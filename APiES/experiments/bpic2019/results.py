import matplotlib.pyplot as plt
import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from APiES.constants import themes
from functions.dataframe_operations import import_df
from APiES.experiments.bpic2019.combine import fn_combined_results_time_mean
from APiES.constants import CLUSTERING_PARAMETER_TYPE, MODEL_TYPE, CLUSTERING_METHOD, CLUSTERING_PARAMETER
from APiES.data_handlers.abstract_gcs_settings import AbstractGCSSettings


def get_colour(mt, cm, cpt):
    if cm == AbstractGCSSettings.NATIVE:
        return themes.SKY_BLUE
    elif cm == AbstractGCSSettings.RANDOM:
        return themes.BRIGHT_ORANGE
    else:
        raise NotImplementedError


def make_title(cpt, mt, cm):
    if cpt == AbstractGCSSettings.CLUSTER_SIZE:
        cpt = r'$\rho$'
    elif cpt == AbstractGCSSettings.CLUSTER_COUNT:
        cpt = r'$k$'

    if mt == 'TestModel':
        mt = 'Test'

    if cm == AbstractGCSSettings.RANDOM:
        cm = 'random'
    elif cm == AbstractGCSSettings.NATIVE:
        cm = 'native'

    return f'{cpt}, {mt}, {cm}'


def get_label_thing(mt, cm, cpt, **kwargs):
    s = []
    if MODEL_TYPE not in kwargs:
        s.append(mt)
    if CLUSTERING_METHOD not in kwargs:
        s.append({AbstractGCSSettings.NATIVE: '$k$-medoids', AbstractGCSSettings.RANDOM: 'random'}[cm])
    if CLUSTERING_PARAMETER_TYPE not in kwargs:
        # s.append(cpt)
        pass

    return ', '.join(s)


def get_parameter_label(single, mt, cm, cpt, **kwargs):
    if not single:
        return ''
    else:
        return get_label_thing(mt, cm, cpt, **kwargs)


def get_single_cluster_label(single, mt, cm, cpt, **kwargs):
    p1 = 'single cluster'
    if not single:
        return p1
    else:
        return f'{p1} ({get_label_thing(mt, cm, cpt, **kwargs)})'


def get_single_entity_label(single, mt, cm, cpt, **kwargs):
    p1 = 'single entity'
    if not single:
        return p1
    else:
        return f'{p1} ({get_label_thing(mt, cm, cpt, **kwargs)})'


def plot(fn_in, single=False, fn_out=None, c='case_rmse', add_regions=False, **kwargs):
    # Import full data
    df = import_df(fn_in).set_index('experiment')

    for k, v in kwargs.items():
        df = df[df[k] == v]

    plot_parameters = [MODEL_TYPE, CLUSTERING_METHOD, CLUSTERING_PARAMETER_TYPE]

    df = df.sort_values(plot_parameters + [CLUSTERING_PARAMETER])

    n_plots = len(df[plot_parameters].drop_duplicates())
    if single:
        f, ax_ = plt.subplots()
        n_col, n_row = 1, 1
        axarr = np.array([ax_] * n_plots)
    else:
        n_row, n_col = {8: (2, 4), 4: (2, 2), 2: (2, 1)}.get(n_plots, NotImplementedError)
        # Determine number of plots
        f, axarr = plt.subplots(n_row, n_col)

    # Converts title to nicely formatted title
    def convert(y_label):
        a, b = y_label.split('_', 1)
        if y_label == 'case_rmse':
            return r'Case-$\mathbf{RMSE}$'
        elif y_label == 'cluster_rmse':
            return r'Cluster-$\mathbf{RMSE}$'
        elif b == 'top_decile_accuracy':
            b = 'decile acc'
        elif b == 'top_decile_f1':
            b = 'decile $F_1$'
        elif b == 'r2':
            b = '$r^2$'
        else:
            warnings.warn(f'Unknown {b}')

        return f'{b} [{a}]'

    y_bot = df[c].min() * .9
    y_top = df[c].max() * 1.1

    # Create a plot for each model x clustering parameter type
    for ((model_type, clustering_method, clustering_parameter_type), df_graph), ax in zip(df.groupby(plot_parameters),
                                                                                          axarr.flatten()):

        # Remove None/1 values (special values)
        mask = df_graph['clustering_parameter'].isna() | (df_graph['clustering_parameter'] == 1)
        x = df_graph.loc[~mask, 'clustering_parameter'].astype(int)
        y = df_graph.loc[~mask, c]

        # Plot data
        colour = get_colour(model_type, clustering_method, clustering_parameter_type)

        ax.plot(x, y,
                label=get_parameter_label(single, model_type, clustering_method, clustering_parameter_type, **kwargs),
                color=colour)

        # Add None/1 values (special values)
        df_1 = df_graph.loc[df_graph['clustering_parameter'] == 1, c]
        df_none = df_graph.loc[df_graph['clustering_parameter'].isna(), c]

        # Interpretation of None/1 values depends on cluster size
        if clustering_parameter_type == 'cluster_size':
            df_entity = df_1
            df_single = df_none
            x_entity = 1
            x_single = max(df[CLUSTERING_PARAMETER])
            label_x = r'$\rho$'
        elif clustering_parameter_type == 'cluster_count':
            df_entity = df_none
            df_single = df_1
            x_entity = max(df[CLUSTERING_PARAMETER])
            x_single = 1
            label_x = r'$k$'
        else:
            raise NotImplementedError

        # Plot the None/1 values as horizontal lines
        if len(df_entity) == 1:
            ax.plot([x_entity], [df_entity.iloc[0]], mfc=colour, ms=10, marker='^s'[clustering_method == 'native'],
                    mec='k',
                    ls='',
                    label=get_single_entity_label(single, model_type, clustering_method, clustering_parameter_type,
                                                  **kwargs))
        if len(df_single) == 1:
            ax.plot([x_single], [df_single.iloc[0]], mfc=colour, ms=10, marker='vo'[clustering_method == 'native'],
                    ls='', mec='k',
                    label=get_single_cluster_label(single, model_type, clustering_method, clustering_parameter_type,
                                                   **kwargs))
        # Final make-up

        ax.set_ylabel(convert(c))
        ax.set_xlabel(label_x)

        if not single:
            ax.legend()
            ax.set_ylim(y_bot, y_top)
            ax.set_title(make_title(clustering_parameter_type, model_type, clustering_method))

    if not single:
        # Remove y-labels of non-left axes
        if axarr.ndim == 2 and axarr.shape[1] > 1:
            for ax in axarr[:, 1:].flatten():
                ax.set_ylabel('')

    # Remove x-labels and ticks of non-bottom axes
    # if axarr.shape[0] > 1:
    #     if axarr.ndim == 1:
    #         axx = axarr[:-1].flatten()
    #     else:
    #         axx = axarr[:-1, :].flatten()
    #     for ax in axx:
    #         ax.set_xticks([])
    #         ax.set_xlabel('')

    if single:
        f.set_size_inches(w=8, h=4)
        axarr.flatten()[0].legend(ncol=1)
        if add_regions:
            ax: plt.Axes = axarr.flatten()[0]
            ax.axvline(x=50, ls=':', color='k')
            ax.axvline(x=100, ls=':', color='k')
            y = ax.get_ylim()[0]*0.4 + ax.get_ylim()[1]*0.6
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            ax.text(25, y, r"$\mathtt{I}$", ha="center", va="center", size=10, bbox=bbox_props)
            ax.text(75, y, r"$\mathtt{II}$", ha="center", va="center", size=10, bbox=bbox_props)
            ax.text(50 + 0.5 * ax.get_xlim()[1], y, r"$\mathtt{III}$", ha="center", va="center", size=10, bbox=bbox_props)

    else:
        f.set_size_inches(w=3.5 * n_col + 1, h=4.5 * n_row + 1)

    # Export and show figure

    if fn_out is None:
        fn_out = fn_in.name.replace('.csv', '')

    (Path(fn_in).parent / 'figures').mkdir(exist_ok=True, parents=True)

    plt.savefig(Path(fn_in).parent / 'figures' / f'{fn_out}.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(Path(fn_in).parent / 'figures' / f'{fn_out}.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # combine('rho')
    plot(fn_combined_results_time_mean('rho'), single=True, fn_out='case_rmse', c='case_rmse', **{MODEL_TYPE: 'AHOT'},
         add_regions=True)
    # plot(fn_combined_results_time_mean('rho'), single=True, fn_out='cluster_rmse', c='cluster_rmse',
    #      **{MODEL_TYPE: 'AHOT'})
