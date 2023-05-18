# pylint: disable=unused-variable
# pylint: disable=unused-import
# pylint: disable=abstract-class-instantiated
# pylint: disable=broad-except
import warnings

warnings.filterwarnings("ignore")

import typing as ty
from loguru import logger
import pyrootutils, os
import hydra
from omegaconf import OmegaConf, DictConfig

from rl.pipeline import runner
from rl.resolvers import get_data_name


ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)


OmegaConf.register_new_resolver("get_data_name", get_data_name)


@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT, "configs"),
    config_name="test",  # change using the flag `--config-name`
)
def main(cfg: DictConfig = None) -> ty.Optional[float]:
    """Train model. You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    assert cfg is not None
    pipeline = runner.run(cfg)
    output = pipeline.get_metric_to_optimize()
    return output


if __name__ == "__main__":
    """You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    main()
