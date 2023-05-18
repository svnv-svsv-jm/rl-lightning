from argparse import ArgumentError
import warnings

from rl.options import define_options
from rl.config import get_config, get_tests, get_name_from_ckpt
from rl.utils.helpers import redirect_output_to_file, create_datamodule, create_model, create_trainer, create_callbacks, create_loggers, fast_dev_run
from rl.utils import mem_warning



def setup(
        name: str,
        tag: str = None,
        opt = None,
        **kwargs
    ):
    """Sets up options, datamodule, model and trainer.
    Args:
        name (str):
            Name of the folder where the YAML configuration files are stored.
        tag (str):
            Tag for the run.
    Returns:
        opt, dm, model, trainer
    """
    # Options
    if opt is None:
        opt = get_config(name, **kwargs)
    # Datamodule
    dm = create_datamodule(opt)
    # Model
    model = create_model(dm, opt)
    # Trainer
    trainer = create_trainer(
        opt=opt,
        callbacks=create_callbacks(opt),
        logger=create_loggers(name, tag, tbdir=opt.trainer.default_root_dir),
    )
    return opt, dm, model, trainer


# Train
def train(**kwargs):
    mem_warning(warn_only=True)
    # Set up
    opt, dm, model, trainer = setup(**kwargs)
    # Fast dev run
    fast_dev_run(model, dm)
    # Train
    trainer.fit(
        model=model,
        datamodule=dm,
    )
    # Test
    outputs = trainer.test(
        model=model,
        datamodule=dm,
    )
    return outputs


# Test
def test(**kwargs):
    # If a list of ckpt_paths, then go recursive
    if kwargs.get('tests', None) is not None:
        tests = get_tests(kwargs.get('tests'))
        kwargs.pop('tests', None) # remove to avoid infinite recursion
        n_tests = len(tests)
        for i, ckpt_path in enumerate(tests):
            print(f"\n\n*** ({i}/{n_tests}) ***\n Running test {i}/{n_tests}:\n\t{ckpt_path}\n************\n\n")
            kwargs['ckpt_path'] = ckpt_path
            test(**kwargs)
    # Rename if necessary
    if kwargs.get('ckpt_path', None) is not None:
        kwargs['name'] = get_name_from_ckpt(kwargs.get('ckpt_path'))
    mem_warning()
    # Set up
    opt, dm, model, trainer = setup(**kwargs)
    ckpt_path = opt.trainer.get('resume_from_checkpoint', None)
    if ckpt_path is None:
        raise ArgumentError(f"When validating or testing, a trained model should be provided but {ckpt_path} was found.")
    # Validate
    outputs = trainer.validate(
        model=model,
        datamodule=dm,
        ckpt_path=ckpt_path,
    )
    # Test
    outputs = trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path=ckpt_path,
    )
    return outputs



# ---------
# Main
# ---------
if __name__ == '__main__':
    # Disable annoying warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    # Parse CLI args
    parser = define_options()
    args = parser.parse_args()
    # Redirect output to file
    redirect_output_to_file(**vars(args))
    # Training
    if args.train:
        train(**vars(args))
    else:
        test(**vars(args))