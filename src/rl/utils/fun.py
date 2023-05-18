from __future__ import print_function

import collections
import importlib
import inspect
import os
import re
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image



def class_for_name(
        name: str,
        module_name: str = 'rl.datasets',
        base_name: str = 'rl',
    ):
    """Load a class from string indicating module's name and class' name.
    Args:
        name (str or data.Dataset):
            The name of the module the class is in, for example `rl.data.MNIST`. You can also pass a dataclass, in that case this function will do nothing.
        module_name (str):
            Module from which to load the class defined in the argument `name`.
        base_name (str):
            This argument frees the user from having to type `rl.{*}`, as it is implied we are going to load from the current package. It can be overriden, though. Pass `None` not to use this arg.
    Returns:
        A class.
    """
    # Handle input: prepend `base_name` if not there
    if base_name is not None and base_name != module_name.split(".")[0]:
        module_name = f"{base_name}.{module_name}"
    # Load module
    if isinstance(name, str):
        # Get dataclass from `name`
        class_name = name
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        c = getattr(m, class_name)
    elif isinstance(name, data.Dataset):
        # do nothing
        c = name
    return c


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(f"Name: {name}")
    print(f"Mean: {mean}")


def save_image(image_numpy, image_path):
    #import scipy.misc
    #scipy.misc.imsave(image_path, image)
    image = np.squeeze(image_numpy)
    image_pil = Image.fromarray(image)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(
        ["%s %s" % (method.ljust(spacing), processFunc(str(getattr(object, method).__doc__))) for method in methodList]) )


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        path = paths
        if not os.path.exists(path):
            os.makedirs(path)


def mkdir(*args, **kwargs):
    mkdirs(*args, **kwargs)


def save_opt_to_disk(
        opt,
        filename="opt",
        savedir: str = None,
        **kwargs
        ):
    """Saves opt to YAML file. Opt has to be a struc or Namespace.
    Args:
        opt (struc || Namespace): options for this experiment.
    """
    if savedir is None:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.tag)
    else:
        expr_dir = savedir
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, f"{filename}.yml")
    print(f"Saving to {file_name}...")
    with open(file_name, 'wt') as opt_file:
        for k, v in sorted(opt.__dict__.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))


# ------------------
# printers
# ------------------
def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def print_debug(
        class_name: str,
        key_val: dict = None,
        named_param: list = None,
        **kwargs
    ):
    msg = f"----- [ERROR:{class_name}] -----\nInfo:"
    if key_val is not None:
        for name, val in key_val.items():
            msg += f"\n\t{name}: {val}"
    if named_param is not None:
        for name, param in named_param:
            msg += f"\n\t{name}: {param.shape}"
    print(msg)
