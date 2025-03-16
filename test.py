#!python

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    # load environment variables from `.env` file if it exists
    # recursively searches for `.env` in all folders starting from work dir
    dotenv=True,
)


import dotenv
import hydra
from omegaconf import DictConfig


@hydra.main(version_base='1.1', config_path=f"{root}/configs", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src import test

    # resolve user provided config
    config = utils.resolve_experiment_config(config)
    # Applies optional utilities
    config = utils.extras(config)

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()
