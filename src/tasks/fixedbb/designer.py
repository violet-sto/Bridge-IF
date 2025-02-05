from omegaconf import OmegaConf, DictConfig

from src import utils
from src.datamodules.datasets import Alphabet, DataProcessor
from src.utils import io
from src.utils.config import compose_config as Cfg
from src.models.fixedbb.generator import IterativeRefinementGenerator
from pathlib import Path
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from collections import namedtuple

GenOut = namedtuple(
    'GenOut', 
    ['output_tokens', 'output_scores', 'attentions']
)



def _full_mask(target_tokens, coord_mask, alphabet):
    target_mask = (
        target_tokens.ne(alphabet.padding_idx)  # & mask
        & target_tokens.ne(alphabet.cls_idx)
        & target_tokens.ne(alphabet.eos_idx)
    )
    _tokens = target_tokens.masked_fill(
        target_mask, alphabet.mask_idx
    )
    _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
    return _tokens, _mask


class Designer:
    _default_cfg = Cfg(
        cuda=False,
        generator=Cfg(
            max_iter=1,
            strategy='denoise',  # ['denoise' | 'mask_predict']
            # replace_visible_tokens=False,
            temperature=0,
            eval_sc=False,  
        )
    )

    def __init__(
        self,
        experiment_path,
        cfg: DictConfig = None
    ):
        self.experiment_path = experiment_path
        self.cfg = cfg

        self._initialize()

    def _initialize(self):
        pl_task, exp_cfg = utils.load_from_experiment(
            self.experiment_path)
        self.pl_task = pl_task
        self.exp_cfg = exp_cfg

        self.model = pl_task.model
        self.model.eval()

        if self.cfg.cuda: 
            self._cuda()

        self.alphabet = pl_task.alphabet
        self.data_processor = DataProcessor()

        self.cfg.generator = utils.config.merge_config(
            pl_task.hparams.generator, self.cfg.generator
        )
        self.generator = IterativeRefinementGenerator(
            alphabet=self.alphabet, 
            **self.cfg.generator
        )

        self._structure: dict = None
        self._predictions: list = None

    def print_config(self, print_exp_cfg=False):
        if print_exp_cfg:
            print(f"======= Experiment Config =======")
            print(OmegaConf.to_yaml(self.exp_cfg.resolve()))        

        print(f"======= Designer Config =======")
        print(OmegaConf.to_yaml(self.cfg))        

    def _cuda(self):
        assert torch.cuda.is_available()
        self.model = self.model.cuda()
        self._device = next(self.model.parameters()).device

    def reset(self):
        self._structure = None
        self._predictions = None

    def set_structure(
            self, 
            pdb_path, 
            chain_list=[],
            masked_chain_list=None,
            verbose=False
        ):
        pdb_id = Path(pdb_path).stem

        print(f'loading backbone structure from {pdb_path}.')
        # coords, native_seq = io.load_coords(pdb_path, None)
        self._structure = self.data_processor.parse_PDB(
            pdb_path, 
            input_chain_list=chain_list, 
            masked_chain_list=masked_chain_list
        )

        # self._structure = {
        #     'name': pdb_id,
        #     'coords': coords, 
        #     'seq': native_seq
        # }
        if verbose: return self._structure

    def _featurize(self):
        batch = self.alphabet.featurize(raw_batch=[self._structure])

        if self.cfg.cuda:
            batch = utils.recursive_to(batch, self._device)

        prev_tokens, prev_token_mask = _full_mask(
            batch['tokens'], batch['coord_mask'], self.alphabet
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)
        return batch

    def generate(self, generator_args={}, need_attn_weights=False):
        batch = self._featurize()

        # outputs = self.generator.generate(
        #     model=self.model, 
        #     batch=batch,
        #     need_attn_weights=need_attn_weights,
        #     **generator_args
        # )
        
        outputs = self.pl_task.forward(batch, return_ids=True)

        output_tokens = outputs[0]
        output_tokens = self.alphabet.decode(output_tokens, remove_special=True)

        self._predictions = GenOut(
            output_tokens=output_tokens, 
            output_scores=outputs[1],
            attentions=outputs[2] if need_attn_weights else None
        )
        return self._predictions

    def calculate_metrics(self):
        native_seq = self._structure['seq']

        for prediction in self._predictions.output_tokens:
            rec = np.mean([(a==b) for a, b in zip(native_seq, prediction)])
            print(f"prediction: {prediction}")
            print(f"recovery: {rec}")
            print()
        
        return rec


    def export_attention(self, saveto, layer=-1, average_heads=False):
        assert self._predictions is not None
        attentions = self._predictions.attentions[-1]

        from bertviz import model_view, head_view

        tokens = self.alphabet.decode(attentions['input'][None], return_as='list')[0]
        attns = attentions['attn_weights'].split(1, dim=0)
        num_layers = len(attns)

        if layer != 'all':
            layer = (num_layers + layer) % num_layers
            saveto = f"{saveto}_l{layer}"
            attns = [attns[layer]]
        if average_heads:
            attns = [attn.mean(dim=1, keepdims=True) for attn in attns]

        html = model_view(attns, tokens, html_action='return')

        with open(saveto + '.html', 'w') as f:
            f.write(html.data)

    def inpaint(self, start_ids, end_ids, generator_args={}, need_attn_weights=False):
        batch = self.alphabet.featurize(raw_batch=[self._structure])
        if self.cfg.cuda:
            batch = utils.recursive_to(batch, self._device)

        prev_tokens = batch['tokens'].clone()
        for sid, eid in zip(start_ids, end_ids):
            prev_tokens[..., sid:eid+1] = self.alphabet.mask_idx

        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx) 

        outputs = self.generator.generate(
            model=self.model, 
            batch=batch,
            need_attn_weights=need_attn_weights,
            replace_visible_tokens=True,
            **generator_args
        )
        output_tokens = outputs[0]

        original_segments = []
        designed_segments = []
        for sid, eid in zip(start_ids, end_ids): 
            original_segment = self.alphabet.decode(
                batch['tokens'][..., sid:eid+1].clone(), remove_special=False)
            original_segments.append(original_segment)

            designed_segment = self.alphabet.decode(
                output_tokens[..., sid:eid+1].clone(), remove_special=False)
            designed_segments.append(designed_segment)

        output_tokens = self.alphabet.decode(output_tokens, remove_special=True)
        self._predictions = GenOut(
            output_tokens=output_tokens, 
            output_scores=outputs[1],
            attentions=outputs[2] if need_attn_weights else None
        )
        return self._predictions, original_segments, designed_segments