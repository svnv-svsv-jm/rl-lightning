import pytest
import os
import pyrootutils
import typing as ty
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from helpers import set_cfg_defaults


ROOT = pyrootutils.find_root(
    search_from=os.path.dirname(__file__),
    indicator=[".git", "pyproject.toml", "setup.py"],
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Define CL args."""
    parser.addoption("--all", action="store_true", default=False)
    parser.addoption("--use-gpu", action="store_true", default=False)


@pytest.fixture(scope="session")
def all_tests(request: pytest.FixtureRequest) -> bool:
    """Whether to perform all tests or not."""
    all_tests = request.config.getoption("--all")
    return bool(all_tests)


@pytest.fixture(scope="session")
def use_gpu(request: pytest.FixtureRequest) -> bool:
    """Whether to test on GPU or not."""
    use_gpu: bool = request.config.getoption("--use-gpu")
    return use_gpu


@pytest.fixture(scope="session")
def data_path() -> str:
    """Path where to download any dataset."""
    return os.environ.get("PATH_DATASETS", "./.data")


@pytest.fixture(scope="session")
def artifacts_location() -> str:
    """Path where to save artifacts."""
    return os.environ.get("PYTEST_ARTIFACTS", "./pytest_artifacts")


@pytest.fixture(scope="session")
def artifact_location(artifacts_location: str) -> str:
    """Path where to save artifacts."""
    return artifacts_location


@pytest.fixture(scope="session")
def checkpoints_path() -> str:
    """Path where to save checkpoints."""
    return os.environ.get("PATH_CHECKPOINTS", "./checkpoints")


@pytest.fixture(scope="session")
def batch_size() -> int:
    """Batch size for tests."""
    return 64 if torch.cuda.is_available() else 32


@pytest.fixture(scope="session")
def config_path() -> str:
    """Configuration path."""
    _config_path = os.path.join("..", "configs")
    # assert os.path.exists(_config_path), f"Error: {_config_path} not found."
    return _config_path


@pytest.fixture(scope="package")
def cfg_hydra_default_2(config_path: str) -> DictConfig:
    """Load MACE global config."""
    with initialize(version_base="1.2", config_path=config_path):
        cfg = compose(
            config_name="test-2.yaml",
            return_hydra_config=True,
            overrides=[
                "tag=mace",
            ],
        )
        dictcfg: DictConfig = set_cfg_defaults(cfg)
        return dictcfg


@pytest.fixture(scope="function")
def cfg_hydra_test_2(cfg_hydra_default: DictConfig) -> ty.Generator:
    """This is called by each test which uses this fixture, each test generates its own temporary logging path."""
    cfg = cfg_hydra_default.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = None
        cfg.paths.log_dir = None
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope="package")
def cfg_hydra_default(config_path: str) -> DictConfig:
    """Load test global config."""
    with initialize(version_base="1.2", config_path=config_path):
        cfg = compose(
            config_name="test.yaml",
            return_hydra_config=True,
            overrides=[
                "tag=test",
                # "paths.output_dir=lightning_logs/pytest",
            ],
        )
        dictcfg: DictConfig = set_cfg_defaults(cfg)
        return dictcfg


@pytest.fixture(scope="function")
def cfg_hydra_test(cfg_hydra_default: DictConfig) -> ty.Generator:
    """This is called by each test which uses this fixture, each test generates its own temporary logging path."""
    cfg = cfg_hydra_default.copy()
    yield cfg
    GlobalHydra.instance().clear()
