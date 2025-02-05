import os
from typing import Any, Callable, List, Union
from pathlib import Path
import numpy as np
import torch
from src import utils
from src.models.fixedbb.generator import IterativeRefinementGenerator, maybe_remove_batch_dim
from src.modules import metrics, noise_schedule, diffusion_utils, cross_entropy
from src.tasks import TaskLitModule, register_task
from src.utils.config import compose_config as Cfg, merge_config

from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric

from tqdm import tqdm

from src.datamodules.datasets.data_utils import Alphabet

# import esm

log = utils.get_logger(__name__)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('fixedbb/mb_pifold')
class MarkovBridge(TaskLitModule):

    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='no_noise',  # ['full_mask', 'random_mask']
            use_context=False,
            num_unroll=0,
        ),
        generator=Cfg(
            max_iter=1,
            strategy='denoise',  # ['denoise' | 'mask_predict']
            noise='full_mask',  # ['full_mask' | 'selected mask']
            replace_visible_tokens=False,
            temperature=0,
            eval_sc=False,
        ),
        version=Cfg(
            dataset='cath_4.2',
        ),
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        alphabet: DictConfig,
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
        generator=_DEFAULT_CFG.generator,
        version=_DEFAULT_CFG.version
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)

        self.alphabet = Alphabet(**alphabet)
        self.build_model() 
        self.build_generator()

        # Diffusion
        self.T = self.hparams.generator.diffusion_steps
        self.noise_schedule = noise_schedule.PredefinedNoiseScheduleDiscrete(
            noise_schedule=self.hparams.generator.diffusion_noise_schedule,
            timesteps=self.T
        )
        self.transition_model = noise_schedule.InterpolationTransition(
            x_classes=len(self.alphabet),
        )
        self.use_context = self.hparams.learning.use_context

        # Load structure encoder
        if version == 'cath_4.2':
            self.load_encoder_from_ckpt('./ckpts/cath_4.2/lm_design_esm1b_650m_pifold/checkpoints/best.ckpt')
        elif version == 'cath_4.3':
            self.load_encoder_from_ckpt('./ckpts/cath_4.3/lm_design_esm1b_650m_pifold/checkpoints/best.ckpt')
        for param in self.model.encoder.parameters():
            param.requires_grad_(False)

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_generator(self):
        self.hparams.generator = merge_config(
            default_cfg=self._DEFAULT_CFG.generator,
            override_cfg=self.hparams.generator
        )
        self.generator = IterativeRefinementGenerator(
            alphabet=self.alphabet,
            **self.hparams.generator
        )
        log.info(f"Generator config: {self.hparams.generator}")

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.alphabet.padding_idx
        self.criterion_ce = cross_entropy.Coord2SeqCrossEntropyLoss(label_smoothing=0.0, ignore_index=1)
        self.criterion_ce.ignore_index = self.alphabet.padding_idx
        
    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()

        self.acc = MeanMetric()
        self.acc_best = MaxMetric()

        self.acc_median = CatMetric()
        self.acc_median_best = MaxMetric()

    def load_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
    
    def load_encoder_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        encoder_state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}

        missing, unexpected = self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_epoch_start(self) -> None:
        if self.hparams.generator.eval_sc:
            import esm
            log.info(f"Eval structural self-consistency enabled. Loading ESMFold model...")
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(self.device)

    # -------# Training #-------- #
    @torch.no_grad()
    def inject_noise(self, tokens, coord_mask, noise=None, sel_mask=None, mask_by_unk=False):
        padding_idx = self.alphabet.padding_idx
        if mask_by_unk:
            mask_idx = self.alphabet.unk_idx
        else:
            mask_idx = self.alphabet.mask_idx

        def _full_mask(target_tokens):
            target_mask = (
                target_tokens.ne(padding_idx)  # & mask
                & target_tokens.ne(self.alphabet.cls_idx)
                & target_tokens.ne(self.alphabet.eos_idx)
            )
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & coord_mask
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        def _adaptive_mask(target_tokens):
            raise NotImplementedError

        noise = noise or self.hparams.noise

        if noise == 'full_mask':
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':
            masked_tokens = tokens
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        prev_tokens = masked_tokens
        prev_token_mask = prev_tokens.eq(mask_idx) & coord_mask
        # target_mask = prev_token_mask & coord_mask

        return prev_tokens, prev_token_mask  # , target_mask

    def apply_noise(self, X, X_T, node_mask):
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        # s_int = t_int - 1

        t_float = t_int / self.T
        # s_float = s_int / self.T

        # beta_t (1-alpha_t) and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        # alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        # Qtb = self.transition_model.get_Qt_bar(
        #     alpha_bar_t=alpha_t_bar,
        #     X_T=self.alphabet.one_hot(X_T), # (bs, n, dx_in)
        #     node_mask=node_mask,
        #     device=self.device,
        # )  # (bs, n, dx_in, dx_out)

        # assert len(Qtb.shape) == 4
        # assert (abs(Qtb.sum(dim=3) - 1.) < 1e-4).all(), Qtb.sum(dim=3) - 1

        # probX = (self.alphabet.one_hot(X).unsqueeze(-2) @ Qtb).squeeze(-2)  # (bs, n, dx_out)
        # X_t = diffusion_utils.sample_discrete_features(probX=probX)
        
        xt_eq_x0_mask = torch.bernoulli(alpha_t_bar.repeat((1, X.shape[-1]))).int()
        X_t = xt_eq_x0_mask * X + (1 - xt_eq_x0_mask) * X_T
        
        assert (X.shape == X_t.shape)

        noisy_data = {
            't_int': t_int,
            't': t_float,
            'beta_t': beta_t,
            # 'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': X_t,
            'node_mask': node_mask,
            'xt_eq_x0_mask': xt_eq_x0_mask
        }
        return noisy_data

    def step(self, batch, batch_idx):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        coords = batch['coords']
        coord_mask = batch['coord_mask']
        tokens = batch['tokens'] # {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3}
        batch_converter = self.alphabet._alphabet.get_batch_converter()

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, coord_mask, noise=self.hparams.learning.noise) # full_mask
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = label_mask = prev_token_mask        

        # 1) generate initial prediction
        encoder_logits, encoder_out = self.model.encoder(batch, return_feats=True)

        encoder_out['feats'] = encoder_out['feats'].detach()

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        ###### align pifold and esm ######
        seqs = self.alphabet.decode(tokens, remove_special=True)
        aligned_tokens = batch_converter([('seq', seq) for seq in seqs])[-1].to(tokens)
        aligned_label_mask = (
                aligned_tokens.ne(1)  # & mask
                & aligned_tokens.ne(0)
                & aligned_tokens.ne(2)
            )
        encoder_out['aligned_feats'] = torch.zeros(aligned_tokens.shape[0],aligned_tokens.shape[1],self.hparams.model.encoder.d_model).to(encoder_out['feats'])
        encoder_out['aligned_feats'][aligned_label_mask] = encoder_out['feats'][coord_mask]
        encoder_out['aligned_label_mask'] = aligned_label_mask
        
        init_seqs = self.alphabet.decode(init_pred, remove_special=True)
        aligned_init_pred = batch_converter([('seq', seq) for seq in init_seqs])[-1].to(init_pred)
        ##################################
        
        # 2) train bridge
        # 2.1: Getting noisy data
        noisy_data = self.apply_noise(
            X=aligned_init_pred, 
            X_T=aligned_tokens, 
            node_mask=aligned_label_mask
            )
        xt_eq_x0_mask = noisy_data['xt_eq_x0_mask']
        # 2.2: Making predictions
        context = aligned_init_pred.clone() if self.use_context else None
        logits = self.model.decoder(
            tokens=noisy_data['X_t'], 
            alpha_t_bar=noisy_data['alpha_t_bar'],
            context=context,
            timesteps=noisy_data['t_int'],
            encoder_out=encoder_out,
        )['logits']

        if isinstance(logits, tuple):
            logits, encoder_logits = logits
            # loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)
            # NOTE: use fullseq loss for pLM prediction
            loss, logging_output = self.criterion(
                logits, tokens,
                # hack to calculate ppl over coord_mask in test as same other methods
                #? label_mask=label_mask if self.stage == 'test' else None
                label_mask=label_mask 
            )
            encoder_loss, encoder_logging_output = self.criterion(encoder_logits, tokens, label_mask=label_mask)

            loss = loss + encoder_loss
            logging_output['encoder/nll_loss'] = encoder_logging_output['nll_loss']
            logging_output['encoder/ppl'] = encoder_logging_output['ppl']
        else:
            if self.hparams.learning.reparam:   
                loss, logging_output = self.criterion_ce(logits, aligned_tokens, label_mask=(xt_eq_x0_mask & aligned_label_mask))
            else:
                loss, logging_output = self.criterion_ce(logits, aligned_tokens, label_mask=aligned_label_mask)
                
            # loss, logging_output = self.compute_training_VLB(aligned_tokens, logits, aligned_label_mask, noisy_data, batch_idx)
            
        return loss, logging_output
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch, batch_idx)

        # log train metrics
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def compute_training_CE_loss_and_metrics(self, true, pred, batch_idx):
        pass

    def compute_training_VLB(self, true, pred, node_mask, noisy_data, batch_idx):
        bsz = true.shape[0]

        n_tokens = true.numel()
        if self.criterion.ignore_index is not None:
            n_nonpad_tokens = true.ne(self.criterion.ignore_index).float().sum()
        sample_size = node_mask.sum()

        z_t = noisy_data['X_t']
        z_T_true = true
        z_T_pred = pred
        t = noisy_data['t_int'] + 1

        true_pX = self.compute_q_zs_given_q_zt(self.alphabet.one_hot(z_t), self.alphabet.one_hot(z_T_true), node_mask, t=t)
        pred_pX = self.compute_p_zs_given_p_zt(self.alphabet.one_hot(z_t), z_T_pred, node_mask, t=t)

        loss = self.criterion(
            masked_pred_X=pred_pX,
            true_X=true_pX,
            label_mask=node_mask
        )

        logging_output = {
            'loss_sum': loss.data,
            'bsz': bsz,
            'sample_size': self.criterion.node_loss.total_samples,
            'sample_ratio': self.criterion.node_loss.total_samples / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens
        }
        self.criterion.reset()
        return loss, logging_output

    # -------# Evaluating #-------- #
    def on_test_epoch_start(self) -> None:
        self.hparams.noise = 'full_mask'

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch, batch_idx)

        # log other metrics
        sample_size = logging_output['sample_size']
        self.eval_loss.update(loss, weight=sample_size)
        # self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        if self.stage == 'fit':
            pred_outs = self.predict_step(batch, batch_idx)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)
            self.log("val/ppl_best", self.val_ppl_best.compute(), on_epoch=True, prog_bar=True)

            self.predict_epoch_end(results=None)

        super().on_validation_epoch_end()

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        # In testing, remove target tokens to ensure no data leakage!
        # or you can just use the following one if you really know what you are doing:
        tokens = batch['tokens']
        # tokens = batch.pop('tokens')

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, batch['coord_mask'],
            noise=self.hparams.generator.noise,  # NOTE: 'full_mask' by default. Set to 'selected_mask' when doing inpainting.
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        output_tokens, output_scores, logits, history = self.sample(
            batch=batch, alphabet=self.alphabet, 
            max_iter=self.T,
            strategy=self.hparams.generator.strategy,
            replace_visible_tokens=self.hparams.generator.replace_visible_tokens,
            temperature=self.hparams.generator.temperature
        )
        if not return_ids:
            return self.alphabet.decode(output_tokens)
        return output_tokens, logits, history
    
    @torch.no_grad()
    def sample(self, batch, alphabet=None, 
               max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
               need_attn_weights=False):
        alphabet = alphabet or self.alphabet
        padding_idx = alphabet.padding_idx
        mask_idx = alphabet.mask_idx

        max_iter = max_iter
        strategy = strategy
        temperature = temperature

        # 0) encoding
        encoder_out = self.model.forward_encoder(batch)

        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = self.model.initialize_output_tokens(
            batch, encoder_out=encoder_out)
        
        ###### align pifold and esm ######
        batch_converter = alphabet._alphabet.get_batch_converter()
        init_seqs = self.alphabet.decode(initial_output_tokens, remove_special=True)
        initial_output_tokens = batch_converter([('seq', seq) for seq in init_seqs])[-1].to(initial_output_tokens)
               
        aligned_label_mask = (
            initial_output_tokens.ne(1)  # & mask
            & initial_output_tokens.ne(0)
            & initial_output_tokens.ne(2)
        )
        encoder_out['aligned_feats'] = torch.zeros(initial_output_tokens.shape[0],initial_output_tokens.shape[1],self.hparams.model.encoder.d_model).to(encoder_out['feats'])
        encoder_out['aligned_feats'][aligned_label_mask] = encoder_out['feats'][batch['coord_mask']]
        encoder_out['aligned_label_mask'] = aligned_label_mask
        ###### align pifold and esm ######
        
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=torch.zeros_like(initial_output_tokens).float(),
            logits=torch.zeros_like(initial_output_tokens).float().unsqueeze(-1).repeat(1,1,33),
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            xt_neq_xT=torch.full_like(initial_output_tokens, True, dtype=torch.bool)
        )

        context = initial_output_tokens.clone() if self.use_context else None

        if need_attn_weights:
            attns = [] # list of {'in', 'out', 'attn'} for all iteration

        # if strategy == 'discrete_diffusion':
        #     prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(batch['prev_tokens'])

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(range(0, max_iter)):
            s_array = s_int * torch.ones((batch['prev_tokens'].size(0), 1)).type_as(batch['coords'])
            t_array = s_array + 1
            # s_norm = s_array / max_iter
            # t_norm = t_array / max_iter

            # Sample z_s
            sampled_s, output_scores, logits, new_xt_neq_xT = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                prev_decoder_out=prev_decoder_out,
                X=initial_output_tokens,
                node_mask=aligned_label_mask, # batch['prev_token_mask']
                context=context,
                encoder_out=encoder_out,
                argmax_decoding=True # True is better
            )

            if replace_visible_tokens:
                visible_token_mask = ~batch['prev_token_mask']
                visible_tokens = batch['prev_tokens']
                output_tokens = torch.where(
                    visible_token_mask, visible_tokens, output_tokens)

            if need_attn_weights:
                attns.append(
                    dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
                         output=maybe_remove_batch_dim(output_tokens),
                         attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
                )

            prev_decoder_out.update(
                output_tokens=sampled_s,
                output_scores=output_scores,
                logits=logits,
                step=self.T-s_int,
                xt_neq_xT=new_xt_neq_xT,
                # history=prev_decoder_out['output_tokens']
            )
            prev_decoder_out['history'].append(sampled_s)

        decoder_out = prev_decoder_out

        if need_attn_weights:
            return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], attns
        return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], prev_decoder_out['history']
    
    def sample_p_zs_given_zt(self, s, t, prev_decoder_out, X, node_mask, context=None, encoder_out=None, argmax_decoding=True):
        # Hack: in direct MB we consider flipped time flow
        X_t = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        xt_neq_xT = prev_decoder_out['xt_neq_xT']
        bs, n = X_t.shape[:2]
        # t = 1 - t
        beta_t = self.noise_schedule(t_int=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 't_int': s, 'node_mask': node_mask}

        pred = self.model.decoder(
            tokens=noisy_data['X_t'], 
            alpha_t_bar=alpha_s_bar,
            context=context,
            timesteps=noisy_data['t_int'], 
            encoder_out=encoder_out)['logits']
        
        # Reperameterized sampling
        scores = torch.softmax(pred, dim=-1)  # bs, n, d0
        if argmax_decoding:
            cur_scores, cur_tokens = scores.max(-1)
            cur_entropy = -torch.distributions.Categorical(probs=scores).entropy()
            # pred_X = pred.argmax(dim=-1)
        else: #TODO for higher diversity
            cur_tokens = torch.distributions.Categorical(logits=pred / 0.1).sample() # bs, n
            cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1) # bs, n

        lowest_k_mask = self.skeptical_unmasking(cur_scores, node_mask, t=s, rate_schedule='linear', topk_mode='deterministic')    
            
        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that 
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.
        
        # #### the `uncond` mode ####
        # In our reparameterized decoding, 
        # both v1_t and v2_t can be fully determined by the current token scores .
        
        not_v1_t = lowest_k_mask # the `uncond` mode
        # not_v1_t = (cur_tokens == X_t) & (cur_scores < output_scores) & lowest_k_mask # the `cond` mode
        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask    
        
        ######## skeptical decoding ########
        masked_to_xT = ~not_v2_t
        masked_to_xT = (masked_to_xT & node_mask).int()
        X_s = (1 - masked_to_xT) * X_t + masked_to_xT * cur_tokens
        
        new_xt_neq_xT = xt_neq_xT
                
        ####### Vanilla ########
        # masked_to_xT = torch.bernoulli(beta_t.repeat((1, X_t.shape[-1]))).int()
        # masked_to_xT = (masked_to_xT & node_mask).int()
        # X_s = (1 - masked_to_xT) * X_t + masked_to_xT * cur_tokens
        
        # new_xt_neq_xT = xt_neq_xT
        
        assert (X_t.shape == X_s.shape)

        return (
            X_s.type_as(X_t),
            cur_scores,
            pred,
            new_xt_neq_xT
        )

    def skeptical_unmasking(self, cur_scores, label_mask, t, rate_schedule, topk_mode='deterministic'):
        # first set the denoising rate according to the schedule
        if rate_schedule == "linear":
            rate = 1 - (t + 1) / self.T
        elif rate_schedule == "cosine":
            rate = torch.cos((t + 1) / self.T * np.pi * 0.5)
        elif rate_schedule == "beta":
            rate = 1 - self.noise_schedule(t_int=t+1)
        else:
            raise NotImplementedError
        
        # compute the cutoff length for denoising top-k positions    
        cutoff_len = (
            label_mask.sum(1, keepdim=True).type_as(cur_scores) * rate
            ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~label_mask, 1000.0)
        
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
            _scores = _scores_for_topk + noise_scale * rate * gumbel_noise
        elif topk_mode == "deterministic":
            _scores = _scores_for_topk
        sorted_scores = _scores.sort(-1)[0]
        cutoff = sorted_scores.gather(dim=-1, index=cutoff_len) + 1e-10
        # cutoff_len = k -> select k + 1 tokens
        masking = _scores < cutoff
        
        return masking

    def compute_q_zs_given_q_zt(self, X_t, X_T, node_mask, t):
        # Hack: in direct MB we consider flipped time flow
        bs, n = X_t.shape[:2]
        beta_t = self.noise_schedule(t_int=t)  # (bs, 1)

        # Compute transition matrices given prediction
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=X_T,
            node_mask=node_mask,
            device=self.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        # Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt  # bs, n, 1, d_t
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # bs, n, d_t
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X

    def compute_p_zs_given_p_zt(self, z_t, pred, node_mask, t):
        p_X_T = F.softmax(pred, dim=-1)  # bs, n, d

        prob_X = torch.zeros_like(p_X_T)  # bs, n, d

        for i in range(len(self.alphabet)):
            X_T_i = self.alphabet.one_hot(torch.ones_like(p_X_T[..., 0]).long() * i)
            z_T = X_T_i
            prob_X_i = self.compute_q_zs_given_q_zt(z_t, z_T, node_mask, t)  # bs, n, d
            prob_X += prob_X_i * p_X_T[..., i].unsqueeze(-1)  # bs, n, d

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        coord_mask = batch['coord_mask']
        tokens = batch['tokens']

        pred_tokens, logits, history = self.forward(batch, return_ids=True)

        batch_converter = self.alphabet._alphabet.get_batch_converter()
        seqs = self.alphabet.decode(tokens, remove_special=True)
        tokens = batch_converter([('seq', seq) for seq in seqs])[-1].to(tokens)
               
        coord_mask = (
            tokens.ne(1)  # & mask
            & tokens.ne(0)
            & tokens.ne(2)
        )
        
        batch['aligned_coords'] = torch.zeros(tokens.shape[0], tokens.shape[1], 4, 3).to(batch['coords'])
        batch['aligned_coords'][coord_mask] = batch['coords'][batch['coord_mask']]
        
        # NOTE: use esm-1b to refine
        # pred_tokens = self.esm_refine(
        #     pred_ids=torch.where(coord_mask, pred_tokens, prev_tokens))
        # # decode(pred_tokens[0:1], self.alphabet)

        if log_metrics:
            # per-sample accuracy
            recovery_acc_per_sample = metrics.accuracy_per_sample(pred_tokens, tokens, mask=coord_mask)
            self.acc_median.update(recovery_acc_per_sample)

            # # global accuracy
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=coord_mask)
            self.acc.update(recovery_acc, weight=coord_mask.sum())

        results = {
            'pred_tokens': pred_tokens,
            'names': batch['names'],
            'native': batch['seqs'],
            'recovery': recovery_acc_per_sample,
            'sc_tmscores': np.zeros(pred_tokens.shape[0])
        }

        if self.hparams.generator.eval_sc:
            torch.cuda.empty_cache()
            sc_tmscores, mean_plddt, pdb_results = self.eval_self_consistency(pred_tokens, batch['aligned_coords'], mask=tokens.ne(self.alphabet.padding_idx))
            results['sc_tmscores'] = sc_tmscores
            results['mean_plddt'] = mean_plddt
            results['pdb_results'] = pdb_results

        return results
    
    def predict_epoch_end(self, results: List[Any]) -> None:
        log_key = 'test' if self.stage == 'test' else 'val'

        acc = self.acc.compute() * 100
        self.acc.reset()
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        acc_median = torch.median(self.acc_median.compute()) * 100
        self.acc_median.reset()
        self.log(f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.acc_best.update(acc)
            self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)

            self.acc_median_best.update(acc_median)
            self.log(f"{log_key}/acc_median_best", self.acc_median_best.compute(), on_epoch=True, prog_bar=True)
        else:
            if self.hparams.generator.eval_sc:
                import itertools
                sc_tmscores = list(itertools.chain(*[result['sc_tmscores'] for result in results]))
                mean_plddt = list(itertools.chain(*[result['mean_plddt'] for result in results]))
                self.log(f"{log_key}/sc_tmscores", np.mean(sc_tmscores), on_epoch=True, prog_bar=True)
                self.log(f"{log_key}/mean_plddt", np.mean(mean_plddt), on_epoch=True, prog_bar=True)
            self.save_prediction(results, saveto=f'./test_tau{self.hparams.generator.temperature}.fasta')

    def save_prediction(self, results, saveto=None):
        save_dict = {}
        if saveto:
            saveto = os.path.abspath(saveto)
            log.info(f"Saving predictions to {saveto}...")
            fp = open(saveto, 'w')
            fp_native = open('./native.fasta', 'w')

        for entry in results:
            for name, prediction, native, recovery, scTM in zip(
                entry['names'],
                self.alphabet.decode(entry['pred_tokens'], remove_special=True),
                entry['native'],
                entry['recovery'],
                entry['sc_tmscores'],
            ):
                save_dict[name] = {
                    'prediction': prediction,
                    'native': native,
                    'recovery': recovery
                }
                if saveto:
                    fp.write(f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | scTM={scTM:.2f}\n")
                    fp.write(f"{prediction}\n\n")
                    fp_native.write(f">name={name}\n{native}\n\n")   
        
        #NOTE for PDB 
        # for entry in results:
        #     for name, prediction, native, recovery, scTM, pdb in zip(
        #         entry['names'],
        #         self.alphabet.decode(entry['pred_tokens'], remove_special=True),
        #         entry['native'],
        #         entry['recovery'],
        #         entry['sc_tmscores'],
        #         entry['pdb_results'],
        #     ):
        #         save_dict[name] = {
        #             'prediction': prediction,
        #             'native': native,
        #             'recovery': recovery
        #         }
        #         if saveto:
        #             fp.write(f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | scTM={scTM:.2f}\n")
        #             fp.write(f"{prediction}\n\n")
        #             fp_native.write(f">name={name}\n{native}\n\n")

        #         with open("./predicted_pdb_initial/{}.pdb".format(name), "w") as f:
        #             f.write(pdb)                  

        if saveto:
            fp.close()
            fp_native.close()
        return save_dict

    def esm_refine(self, pred_ids, only_mask=False):
        """Use ESM-1b to refine model predicted"""
        if not hasattr(self, 'esm'):
            import esm
            self.esm, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            # self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_batcher = self.esm_alphabet.get_batch_converter()
            self.esm.to(self.device)
            self.esm.eval()

        mask = pred_ids.eq(self.alphabet.mask_idx)

        # _, _, input_ids = self.esm_batcher(
        #     [('_', seq) for seq in decode(pred_ids, self.alphabet)]
        # )
        # decode(pred_ids, self.alphabet)
        # input_ids = convert_by_alphabets(pred_ids, self.alphabet, self.esm_alphabet)

        input_ids = pred_ids
        results = self.esm(
            input_ids.to(self.device), repr_layers=[33], return_contacts=False
        )
        logits = results['logits']
        # refined_ids = logits.argmax(-1)[..., 1:-1]
        refined_ids = logits.argmax(-1)
        refined_ids = convert_by_alphabets(refined_ids, self.esm_alphabet, self.alphabet)

        if only_mask:
            refined_ids = torch.where(mask, refined_ids, pred_ids)
        return refined_ids

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def eval_self_consistency(self, pred_ids, positions, mask=None):
        pred_seqs = self.alphabet.decode(pred_ids, remove_special=True)

        # run_folding:
        sc_tmscores = []
        pdb_outputs = []
        with torch.no_grad():
            output = self._folding_model.infer(sequences=pred_seqs, num_recycles=4)
            # pred_seqs = self.alphabet.decode(output['aatype'], remove_special=True)
            for i in range(positions.shape[0]):
                pred_seq = pred_seqs[i]
                seqlen = len(pred_seq)
                _, sc_tmscore = metrics.calc_tm_score(
                    positions[i, 1:seqlen + 1, :3, :].cpu().numpy(),
                    output['positions'][-1, i, :seqlen, :3, :].cpu().numpy(),
                    pred_seq, pred_seq
                )
                sc_tmscores.append(sc_tmscore)
                
                pdb_output = self._folding_model.infer_pdb(pred_seq)
                pdb_outputs.append(pdb_output)
                
        return sc_tmscores, output['mean_plddt'].tolist(), pdb_outputs


def convert_by_alphabets(ids, alphabet1, alphabet2, relpace_unk_to_mask=True):
    sizes = ids.size()
    mapped_flat = ids.new_tensor(
        [alphabet2.get_idx(alphabet1.get_tok(ind)) for ind in ids.flatten().tolist()]
    )
    if relpace_unk_to_mask:
        mapped_flat[mapped_flat.eq(alphabet2.unk_idx)] = alphabet2.mask_idx
    return mapped_flat.reshape(*sizes)
