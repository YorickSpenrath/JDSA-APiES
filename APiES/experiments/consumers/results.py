import itertools
import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from functions import heatmap
from functions.dataframe_operations import import_df
from APiES import constants as sc
from APiES.data_handlers import AbstractDataHandler, AbstractGCSSettings
from APiES.data_handlers.consumer_data_handler import ConsumerDataHandler
from APiES.experiments.consumers.combine import fn_combined_results_global, \
    fn_combined_results_time_mean, all_completed_cdh

fd_figures = sc.CCS_ROOT / 'common' / 'figures'


def add_events(ax, ano=False, label=True, **kwargs):
    x_coordinates = [19.142857142857142, 30.428571428571427, 35.0]
    colours = list('rgb')
    ret = []
    try:
        from ..._cdh_local import event_labels
    except (ModuleNotFoundError, ImportError):
        event_labels = [f'Event {i + 1}' for i in range(3)]

    for i, (d, c, l) in enumerate(zip(x_coordinates, colours, event_labels)):
        kwargs.setdefault('ls', ':')
        if label:
            kwargs['label'] = f'Event {i + 1}' if ano else l
        ret.append(ax.axvline(d, color=c, **kwargs))

    return ret


titles = dict(
    cluster_rmse=r'cluster-$\mathbf{RMSE}$',
    consumer_id_rmse=r'shopper-$\mathbf{RMSE}$',
    consumer_id_turnover=r'$\mathbf{APE}$ on total turnover',
    cluster_turnover=r'cluster-$\mathbf{APE}$ on total turnover',
    consumer_id_turnover_drop_absolute_f1=r'$\mathbf{F_1}$ on interesting shoppers'
)


def create_heatmap(param: str, value_name: str, ot: bool, show_plot: bool = False, filter_kwargs=None):
    """
    Creates a heatmap for the consumer results.

    Parameters
    ----------
    param: str
        'k' or 'rho'
    value_name: str
        The measured value name
    ot: bool
        If True, takes the mean/ci of all time steps. If False, takes the global results
    show_plot: bool
        If True, plot is shown
    filter_kwargs: dict[str -> Callable]
        For every k,v in filter_kwargs, the experiment is filtered on the value of k, using the callable to determine
        whether the experiment should be kept or not.
    """
    if ot:
        df = import_df(fn_combined_results_time_mean(param))
    else:
        df = import_df(fn_combined_results_global(param))

    # Filter on specific results
    if filter_kwargs is not None:
        for k, v in filter_kwargs.items():
            df = df[df[k].apply(v)]

    if param == 'rho':
        assert (df[sc.CLUSTERING_PARAMETER_TYPE] == AbstractGCSSettings.CLUSTER_SIZE).all()

        def convert(x):
            if pd.isna(x):
                return '$|C|$'
            else:
                return f'$2^{{{int(math.log2(x))}}}$'

        x_label = r'$\rho$'

    elif param == 'k':
        assert (df[sc.CLUSTERING_PARAMETER_TYPE] == AbstractGCSSettings.CLUSTER_COUNT).all()

        def convert(x):
            if pd.isna(x):
                return '$|C|$'
            else:
                return str(int(x))

        x_label = '$k$'

    else:
        raise NotImplementedError

    kwargs = {

        'cluster_rmse': dict(inverted=True, fn='cluster_rmse', aspect=.45, value_font_size=6),
        'consumer_id_rmse': dict(inverted=True, fn='shopper_rmse', aspect=.45, value_font_size=6),
        'consumer_id_turnover': dict(inverted=True, fn='turnover_ape', multiply=100, aspect=.45, value_font_size=6),
        'cluster_turnover': dict(inverted=True, fn='cluster_turnover_ape', multiply=100, aspect=.45, value_font_size=6),
        # 'consumer_id_top_decile_f1': dict(inverted=False),
        'consumer_id_turnover_drop_absolute_f1': dict(inverted=False, fn='turnover_drop_f1', aspect=.45,
                                                      value_font_size=6)
    }[value_name]

    fn = kwargs.pop('fn', f'{param}_{"time_" if ot else ""}{value_name}')
    multiply = kwargs.pop('multiply', 1)
    title = titles.get(value_name, value_name)

    def clean_df(dfx):
        dfx = dfx.sort_values('tau')
        dfx = dfx.sort_values(sc.CLUSTERING_PARAMETER, na_position='last', kind='mergesort')
        dfx[sc.CLUSTERING_PARAMETER] = dfx[sc.CLUSTERING_PARAMETER].apply(convert)
        return dfx

    df = clean_df(df)
    df[value_name] = df[value_name] * multiply

    _, ax = heatmap.heatmap_from_raw_data(df, df_ci=None, value_name=value_name,
                                          horizontal_dimension=sc.CLUSTERING_PARAMETER,
                                          vertical_dimension='tau', sort=False,
                                          **kwargs)

    ax.set_ylabel(r'$\tau$')
    ax.set_xlabel(x_label)

    ax.set_title(title)
    plt.savefig(fd_figures / f'{fn}.svg', bbox_inches='tight')
    plt.savefig(fd_figures / f'{fn}.pdf', bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def create_plot(list_of_adh: Iterable[AbstractDataHandler], metric):
    """

    Parameters
    ----------
    list_of_adh: Iterable of AbstractDataHandler
        List of Abstract Data Handlers which to plot
    metric: str
        Metric to plot

    Returns
    -------
    f: plt.Figure
        Figure where the plot is in
    ax: plt.Axes
        Axes where the plot is in
    """
    f, ax = plt.subplots()

    l1, l2 = itertools.tee(list_of_adh)

    def make_label(adh_: AbstractDataHandler):
        z = adh_.settings.clustering_parameter
        if z is None:
            z = '|C|'
        if adh_.settings.clustering_parameter_type == adh_.settings.CLUSTER_SIZE:
            if isinstance(z, int):
                z = f'2^{{{int(math.log2(z))}}}'
            return rf'$\rho={z}$'
        elif adh_.settings.clustering_parameter_type == adh_.settings.CLUSTER_COUNT:
            return rf'$k={z}'
        else:
            raise NotImplementedError

    for adh, colour, marker in zip(l2, sc.themes.NEW_SET + sc.themes.DARK_COLOURS, list('v^osD8')):
        sr = import_df(adh.fn_results_over_time).set_index('metric').loc[metric]
        sr.index = map(int, sr.index)
        ax.plot(sr.index, sr.to_numpy(), color=colour, ls='-', marker=marker, label=make_label(adh))

    add_events(ax=ax, ano=True, label=False)
    ax.set_title(titles.get(metric, metric))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_xlabel('$t$')
    ax.legend(ncol=2)
    return f, ax


def compute_figures(param, **kwargs):
    """
    Computes all relevant heatmaps for the given param

    Parameters
    ----------
    param: str
        'k' or 'rho'

    Other Parameters
    ----------------
    See `py:meth:create_heatmap' for details

    """
    ot = True
    for metric in ['consumer_id_rmse', 'cluster_rmse',
                   'consumer_id_turnover', 'consumer_id_turnover_drop_absolute_f1',
                   'cluster_turnover']:
        create_heatmap(param, metric, ot=ot, **kwargs)


def create_time_f1():
    def f(x: ConsumerDataHandler):
        if x.settings.tau != 9:
            return False
        if x.settings.clustering_parameter not in [1, 2, 2 ** 9, None]:
            return False
        return True

    f, ax = create_plot(filter(f, all_completed_cdh('rho')), 'consumer_id_turnover_drop_absolute_f1')
    ax.set_aspect(8)
    f.set_size_inches(8, 3.5)
    fd_figures.mkdir(exist_ok=True, parents=True)
    plt.savefig(fd_figures / f'time_f1.pdf', bbox_inches='tight')
    plt.savefig(fd_figures / f'time_f1.svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    def filter_rho(x):
        if pd.isna(x):
            return True
        if x == 1:
            return True
        z = int(math.log2(x))
        return z % 2 != 0


    compute_figures('rho', filter_kwargs={sc.CLUSTERING_PARAMETER: filter_rho})
