import time, os, sys, torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, GPUStatsMonitor
from omegaconf.errors import ConfigAttributeError
from pprint import pformat
from tqdm.auto import tqdm

from rl.datasets import DM
from rl.callbacks import *
from rl.utils import class_for_name, Timer, Logger



# Datamodule
def create_datamodule(opt):
    # info
    print(f"---\nCreating dataset {opt.data.dataset} with parameters:")
    for key, val in opt.data.params.items():
        print(f"\t{key}: {val}")
    print("---")
    # kwargs
    kwargs = opt.data
    dataclass_kwargs = opt.data.params.copy()
    kwargs.pop('params', None)
    # create datamodule with provided dataclass
    dm = DM(
        **kwargs,
        dataclass_kwargs=dataclass_kwargs,
    )
    return dm


# Model
def create_model(dm: pl.LightningDataModule, opt):
    # create example input array
    dm.setup(stage='fit')
    example_input_array = dm.create_example_input_array()
    # get model class
    model_class = class_for_name(opt.model.name, "models")
    assert opt.model.name == model_class.__name__, f"Something is wrong: should instantiate a model of class {opt.model.name} but we have {model_class.__name__}...\n"
    # create model instance
    ckpt_path = opt.trainer.get('resume_from_checkpoint', None)
    try:
        if ckpt_path is None:
            # create new model
            print(f"Creating {model_class.__name__} model...\n")
            model = model_class(
                datamodule=dm,
                **dict(opt.training.optimizers),
                **dict(opt.training.checkpoints),
                **dict(opt.model.params),
                example_input_array=example_input_array,
            )
        else:
            # load from checkpoint
            print(f"Loading {model_class} model from {ckpt_path}...\n")
            model = model_class.load_from_checkpoint(ckpt_path)
    except ConfigAttributeError as ex:
        raise Exception(f"This is the current configuration:\n{pformat(opt, indent=2)}") from ex
    return model


# Logger
def create_loggers(
        name: str,
        tag: str = None,
        tbdir: str = "/src/tensorboard",
    ):
    assert isinstance(name, str), f"Argument name must be a string, but a {type(name)} was provided."
    name = str(name)
    # Tagging
    if isinstance(tag, str):
        name += f"-{tag}"
    # TensorBoardLogger
    tb_logger = TensorBoardLogger(
        str(tbdir),
        name=name,
        log_graph=True,
    )
    print(f"\nWill be logging results to '{tbdir}/{name}'.\n")
    return tb_logger


# Callbacks
def create_callbacks(opt, callbacks: list = None):
    callbacks = []
    for cb in opt.callbacks:
        # params to pass: also add 'log_every_n_steps' from the trainer opts
        params = cb.params if ('params' in cb and cb.params is not None) else {}
        if 'log_every_n_steps' in opt.trainer and opt.trainer.log_every_n_steps is not None:
            params['log_every_n_steps'] = opt.trainer.log_every_n_steps
        # cb class
        cb_class = class_for_name(cb.name, module_name='callbacks')
        # cb instance
        cb_obj = cb_class(opt=opt, **params)
        assert isinstance(cb_obj, Callback), f"Wrong type for input `callbacks`: {type(cb_obj)}. Please provide a Callback."
        # add
        callbacks += [cb_obj]
    callbacks += [GPUStatsMonitor()]
    # Info
    print(f"--- Will be using the following callbacks: {[type(x).__name__ for x in callbacks]}")
    # Return
    return callbacks


# Trainer
def fast_dev_run(model, datamodule):
    """Run fast dev so to immediately raise an exception, if any.
    """
    print("\n----------------\n fast_dev_run: start \n----------------\n")
    _debugging = model._debugging
    model._debugging = True
    trainer_dev = pl.Trainer(fast_dev_run=True)
    trainer_dev.fit(
        model=model,
        datamodule=datamodule,
    )
    model._debugging = _debugging
    print("\n----------------\n fast_dev_run: end \n----------------\n")


def create_trainer(
        opt,
        callbacks: list,
        logger: list,
    ):
    """
    Args:
        opt (Namespace):
            Options for this experiment.
        callbacks (list):
            List of PyLit Callbacks to be passed to the Trainer.
        logger (list):
            List of PyLit Loggers to be passed to the Trainer.
    Returns:
        pl.Trainer:
            PyLit Trainer for the experiment.
    """
    # Handle CPU/GPU
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    # Trainer
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        # Callbacks
        callbacks=callbacks,
        # Logging
        logger=logger,
        # Any other
        **opt.trainer,
    )
    return trainer


# Others
def size_info(dm):
    dm.setup(stage='fit')
    # Get useful sizes from sample of data
    with Timer(f"Loading batch"):
        for batch_idx, batch in enumerate(dm.train_dataloader()):
            B = batch[0].size(0)
            for b in range(B):
                print(f"b = {b}")
                for i in range(dm.fit._n):
                    print(f"Modality {i} size: {batch[i][b].size()} [Missing: {not batch[-2][b,0]}]")
                print(f"Label size: {batch[-1][b].size()}")
                break
            break
    return dm


def debug_img_aug_params(dm):
    for attr in dm.fit._img_aug.__dict__:
        val = getattr(dm.fit._img_aug, attr)
        if isinstance(val, (int,float)):
            print(attr, val)


def plot_data(dm):
    dm.setup(stage='fit')
    with Timer(f"Plotting"):
        for batch_idx, batch in enumerate(dm.train_dataloader()):
            B = batch[0].size(0)
            b = np.random.randint(B)
            fig, axs = plt.subplots(1, dm.fit._n)
            fig.suptitle(f"batch_idx={batch_idx}; sample_idx={b}; label={batch[-1][b]}")
            # plot each modality
            for i in range(dm.fit._n):
                if dm.fit._types[i].lower() == "image":
                    n_channels = batch[i][b].size(0)
                    if n_channels > 1:
                        # 3 channles
                        axs[i].imshow(batch[i][b].permute(1,2,0))
                    else:
                        # 1 channel
                        axs[i].imshow(batch[i][b][0])
                elif dm.fit._types[i].lower() == "text":
                    # it's text
                    for j in batch[i][b]:
                        print(dm.fit.get_i2w()[str(int(j))])
            # factors
            if dm.fit.return_factors:
                factors = batch[-3][b]
                factor_names = dm.fit.factor_names
                assert len(factor_names) == factors.size(-1), f"Found {len(factor_names)} factor names but there are {factors.size(-1)} factors...\n{factor_names}: {factors[0]}"
                print(f"Factors for current sample {b}: {factors.size()}")
                for i, name in enumerate(factor_names):
                    print(f"\t{name}: {factors[i]}")
            break


def redirect_output_to_file(
        train: bool = True,
        tests: str = None,
        logdir: str = 'logs',
        name: str = None,
        tag: str = None,
        **kwargs
    ):
    # Get time information and prefix
    timestr = time.strftime("%Y:%m:%d-%H:%M:%S")
    prepend = 'train' if train else 'test'
    # Create filename for output file
    if tests is not None:
        filename = f"{logdir}/tests-{tests}"
    else:
        filename = f'{logdir}/{prepend}-{name}'
    if tag is not None:
        filename += f'-{tag}'
    filename = f'{filename}__{timestr}.log' # output redirected to this file
    if os.path.exists(filename): os.remove(filename) # remove if not exists
    # Create logger and redirect stdout and stderr
    logger = Logger(filename)
    sys.stdout = logger # redirect stdout
    sys.stderr = logger # redirect stderr