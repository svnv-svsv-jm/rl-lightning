'''Script to create and merge the config files according to the chosen dataset (default Mnist)
'''

__all__ = ['get_config', 'get_tests', 'get_name_from_ckpt']


import glob, os
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig



ACCPT_FILES = ('*.yaml', '*.yml')
LOC = os.path.dirname(os.path.realpath(__file__))



def get_config(
        name: str,
        loc: str = LOC,
        accepted_files: tuple = ACCPT_FILES,
        verbose: bool = True,
        ckpt_path: str = None,
        **kwargs
    ):
    """Imports configuration for the current experiment. Parses YAML files in a specified location. Files' extension must be either `yml` or `yaml`, otherwise they must be specified.
    Args:
        name (str):
            Experiment's folder's name from where to load the config files.
        loc (str, optional):
            Location of the config files. Defaults to this file's folder.
        accepted_files (tuple, optional):
            By default, files' extension must be either `yml` or `yaml`. Edit this arg to override.
        verbose (bool, optional): Defaults to False.
        ckpt_path (str, optional):
            Path to a checkpoint. If present, config will be loaded from there. Please note that, even if this argument is not provided but the key `trainer.resume_from_checkpoint` is defined, then config options will be loaded from there, as if this arg is provided.
    Returns:
        opt: Options for training.
    """
    default_opt = get_default_config()
    # If present, just load config from the provided checkpoint path
    if ckpt_path is not None:
        # Reload opts with checkpoint's `hparams.yaml`: this if statement is True if argument `ckpt_path` is passed or options file has `trainer.resume_from_checkpoint`
        ckpt_opt = get_config_from_checkpoint(ckpt_path)
        opt = OmegaConf.merge(default_opt, ckpt_opt)
        opt.trainer['resume_from_checkpoint'] = ckpt_path
        return opt
    # Let's load config options from files: please not that if the key `trainer.resume_from_checkpoint` is defined, then config options will be loaded from there, similarily to if `ckpt_path` was provided
    # Handle deprecated kwargs
    name = str(kwargs.get('dataset', name))
    assert isinstance(name, str), f"Argument name must be a string, but a {type(name)} was provided."
    # Load user config which will override the default config if needed
    files = []
    for ext in accepted_files:
        files += glob.glob(f'{loc}/experiments/{name.lower()}/{ext}')
    if len(files) == 0:
        opt = default_opt
    else:
        # Read config files
        exp_opt = OmegaConf.merge(*[OmegaConf.load(f) for f in files])
        # An empty `trainer` field will raise an error, thus we remove it if so
        if 'trainer' in exp_opt and exp_opt.trainer is None:
            del exp_opt['trainer']
        # Checkpoints
        if (ckpt_path is None) and (exp_opt.get('trainer', None) is not None):
            # If `resume_from_checkpoint` is present, options will be loaded from its `hparams.yaml`
            ckpt_path = exp_opt.trainer.get('resume_from_checkpoint', None)
        if ckpt_path is not None:
            # Reload opts with checkpoint's `hparams.yaml`: this if-statement is True if argument `ckpt_path` is passed or options file has `trainer.resume_from_checkpoint`
            ckpt_opt = get_config_from_checkpoint(ckpt_path)
            ckpt_opt.trainer['resume_from_checkpoint'] = ckpt_path
            opt = OmegaConf.merge(default_opt, ckpt_opt)
        else:
            # Do nothing (keep options from files), but make sure this message is printed instead
            if verbose: print(f"Loading experiment's configuration from: {files}")
            opt = OmegaConf.merge(default_opt, exp_opt)
    assert opt.get('trainer', None) is not None, f"Current configuration does not have a `trainer` field, which is not allowed."
    return opt


def get_default_config(
        loc: str = LOC,
        accepted_files: tuple = ACCPT_FILES,
        verbose: bool = True,
    ):
    """Loads default configuration.
    Args:
        loc (str, optional):
            Location of the config files. Defaults to this file's folder.
        accepted_files (tuple, optional):
            By default, files' extension must be either `yml` or `yaml`. Edit this arg to override.
        verbose (bool, optional): Defaults to False.
    Returns:
        default_opt: default options.
    """
    # Read default config files to make sure everthing is there
    default_files = []
    for ext in accepted_files:
        default_files += glob.glob(f'{loc}/default/{ext}')
    if verbose: print(f"Loading default configuration from: {default_files}")
    default_opt = OmegaConf.merge(*[OmegaConf.load(f) for f in default_files])
    assert default_opt.get('trainer', None) is not None, f"Missing field 'trainer' in configuration. Although PyTorch Lightning's Trainer can also be initiated with any parameter, this is an indication that the default config files are either missing or have been compromised, as they should usually have the `trainer` field."
    return default_opt


def get_config_from_checkpoint(
        ckpt_path: str,
        accepted_files: tuple = ACCPT_FILES,
        verbose: bool = True,
    ):
    """ckpt_path (str): Defaults to None.
            If present, the model will be loaded from the path specified in this argument. This is usually the path to a checkpoint. However, this function will consider that the hparams.yaml file is in `../` w.r.t. the checkpoint file.
        accepted_files (tuple, optional):
            By default, files' extension must be either `yml` or `yaml`. Edit this arg to override.
        verbose (bool, optional): Defaults to False.
    """
    if verbose: print(f"Provided checkpoint path: {ckpt_path}")
    ckpt_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(ckpt_path),
            '..',
        )
    )
    if verbose: print(f"Recognized checkpoint directory: {ckpt_dir}")
    # Load `hparams.yaml` file: load all files ending with eitherr of `*.yml`, `*.yaml`
    hparams_files = []
    for ext in accepted_files:
        hparams_files += glob.glob(f'{ckpt_dir}/{ext}')
    if verbose: print(f"Loading configuration from checkpoint: {hparams_files}")
    # Read config files
    opt = OmegaConf.merge(*[OmegaConf.load(f) for f in hparams_files])
    return opt.opt


def get_tests(
        name: str,
        loc: str = LOC,
        accepted_files: tuple = ACCPT_FILES,
        verbose: bool = True,
    ):
    """Returns a list of checkpoints from specified file.
    """
    # Load YAML file specified in name
    files = []
    for ext in accepted_files:
        files += glob.glob(f'{loc}/tests/{name.lower()}{ext}')
    # Check there is only one match
    assert len(files) == 1, f"{len(files)} files were found for name {name}: {files}. Please make sure that exactly one file is specified."
    # Read config files
    opt = OmegaConf.merge(*[OmegaConf.load(f) for f in files])
    # Sanity check
    assert "tests" in opt.keys(), f"The found YAML config file '{name}' does not have a 'tests' field."
    assert isinstance(opt.tests, (tuple,list,ListConfig)), f"The provided YAML config file '{name}' does not contain a list for its field 'tests' but a {type(opt.tests)}."
    # Print information
    msg = f"Found the following {len(opt.tests)} checkpoint(s) for testing:"
    for t in opt.tests:
        msg += f"\n - {t}"
    print(msg)
    # Return
    return opt.tests


def get_name_from_ckpt(ckpt_path: str):
    return os.path.basename(
        os.path.abspath(
            os.path.join(
                os.path.dirname(ckpt_path),
                '..',
                '..',
            )
        )
    )