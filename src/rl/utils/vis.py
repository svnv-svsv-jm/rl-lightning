from abc import abstractclassmethod
import matplotlib.pyplot as plt
import torch, os, glob
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from matplotlib.colors import BASE_COLORS
from matplotlib.lines import Line2D
from matplotlib import lines



# Constants
COLORS_ALL = [key for key in BASE_COLORS.keys() if key != 'w']
COLORS = ['k', 'r', 'b', 'g', 'c', 'm', 'y']
MARKERS_ALL = [key for key in Line2D.markers.keys() if key not in ['None', None, ' ', '']]
MARKERS = ['*', 'o', 'v', '<', '>', 's', 'p']
LINES = [key for key in lines.lineStyles.keys() if key not in ['None', None, ' ', '']]
LABELS = ['train', 'val', 'test']


def generate_plot_kwargs(
        n: int = 1,
        label = None,
        marker = None,
        linestyle = None,
        color = None,
        **kwargs
    ):
    n_in = n
    if n < 1:
        n = 1
    plot_kwargs = []
    for i, lb, mk, ln, cl in zip(
            range(n),
            itertools.cycle(LABELS),
            itertools.cycle(MARKERS),
            itertools.cycle(LINES),
            itertools.cycle(COLORS),
        ):
        # Generate kwargs
        plot_kwargs += [dict(
            label = label if label is not None else lb,
            marker = marker if marker is not None else mk,
            linestyle = linestyle if linestyle is not None else ln,
            color = color if color is not None else cl,
        )]
    if n_in < 1: plot_kwargs = plot_kwargs[0]
    return plot_kwargs


class TensorboardPlot():
    # NOTE: https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically
    def __init__(self, *files_info: str, **kwargs):
        """Useful for plotting Tensorboard-extracted data. Also check out the classmethod `options_template`.
        Args:
            file_info (str || dict):
                Iterable of CSV file paths; or iterable of dictionaries containing at least the key 'path', plus another nested dictionary of plot arguments at the key 'plot_args'.
            **kwargs:
                If first arg is an iterable of file paths, then you can also pass iterables of plot keyword args. For example: `tb_plot = TensorboardPlot(file1, file2, label=['label1', 'label2'])`.
        """
        if kwargs == {}:
            self.init_from_files_info(*files_info)
        else:
            options = self.init_opt_from_lists(files_info, **kwargs)
            self.init_from_files_info(*options)

    def init_from_files_info(self, *files_info: str):
        # init
        self.data = [None] * len(files_info)
        self.plot_kwargs = [None] * len(files_info)
        default_plot_kwargs = generate_plot_kwargs(len(files_info))
        # read
        for i, opt in enumerate(files_info):
            plot_kwargs = default_plot_kwargs[i]
            if isinstance(opt, str):
                data = pd.read_csv(opt, index_col=0)
            else:
                data = pd.read_csv(str(opt['path']), index_col=0)
                if opt.get('plot_kwargs', None) is not None:
                    for key, val in opt.get('plot_kwargs').items():
                        plot_kwargs[key] = val
            self.data[i] = data[['Step', 'Value']]
            self.plot_kwargs[i] = plot_kwargs

    @abstractclassmethod
    def options_template(cls):
        """Returns a template dict.
        """
        return {
            "path": None,
            "plot_kwargs": generate_plot_kwargs(0),
        }

    def init_opt_from_lists(self, files: list, **kwargs):
        options = []
        default_plot_kwargs = generate_plot_kwargs(len(files))
        for i, f in enumerate(files):
            options += [dict(
                path=f,
                plot_kwargs=default_plot_kwargs[i],
            )]
        for i in range(len(files)):
            for key in options[i]['plot_kwargs'].keys():
                if key in kwargs:
                    options[i]['plot_kwargs'][key] = kwargs[key][i]
        return options

    def rescale(self, i: int, val: float):
        self.data[i]['Value'] = self.data[i]['Value'] * val

    def plot(self,
            savename: str = None,
            title: str = ' ',
            xlabel: str = 'Step [-]',
            ylabel: str = 'Loss [-]',
            log: bool = False,
            a_tol: float = 0,
        ):
        """Creates a plot with curves coming from exported Tensorboad graphs.
        Args:
            savename (str):
                If present, will save the figure with this name.
        """
        fig, axs = plt.subplots(1, 1, figsize=(6,4))
        for i, data in enumerate(self.data):
            axs.plot(data['Step'], data['Value'] + a_tol, **self.plot_kwargs[i])
            axs.legend()
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        axs.set_title(title)
        if log: axs.set_yscale('log')
        if savename is not None:
            if os.path.exists(savename):
                os.remove(savename)
            fig.savefig(savename, dpi=fig.dpi, bbox_inches='tight')
            print(f"Figure saved as {savename}!\n")
        return fig, axs


def choose_files(res_dir: str, pattern: str, after: str, before: str):
    files = glob.glob(f"{res_dir}/{pattern}")
    labels = []
    for f in files:
        filename = os.path.basename(f)
        print(filename)
        start = filename.find(after) + len(after)
        end = filename.find(before)
        labels += [filename[start:end]]
    print("labels: ", labels)
    return files, labels


def imshow_cmap(
        values: torch.Tensor,
        ax: plt.axis,
        fig: plt.figure,
        cmap: str = 'viridis',
        low: float = None,
        high: float = None,
        **kwargs
    ):
    # args
    if low is None or high is None:
        low = np.min(values)
        high = np.max(values)
    # create an axes on the right side of ax. The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ticks = np.linspace(low, high, num=5).tolist()
    im = ax.imshow(values, cmap=cmap, vmin=low, vmax=high)
    cbar = fig.colorbar(im, ax=ax, ticks=ticks, cax=cax)
    cbar.set_ticks(ticks)


def sentence_from_vector(sentence_vector=None, i2w=None):
        sentence= ''
        compt = 0
        for i in sentence_vector:
            compt += 1
            if compt <= 7:
                sentence = sentence + ' ' + (i2w[str(int(i.argmax()))])
            else:
                sentence = sentence + '\n'
                compt = 0 
        return sentence


# get residuals
def get_residuals(X, T):
    tol = 1e-4 * T.abs().max() + 1e-9
    return (T - X / (T + tol)).abs()