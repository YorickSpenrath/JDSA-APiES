import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm


def heatmap_from_raw_data(raw_data: pd.DataFrame, value_name: str, horizontal_dimension: str, vertical_dimension: str,
                          sort=True, df_ci=None, **kwargs):
    df = raw_data.groupby([vertical_dimension, horizontal_dimension], sort=sort)[value_name].mean().unstack()

    if df_ci is not None:
        assert len(raw_data[[vertical_dimension, horizontal_dimension]].drop_duplicates()) == len(raw_data)
        df_ci = df_ci.groupby([vertical_dimension, horizontal_dimension], sort=sort)[value_name].mean().unstack()

    return make_heatmap(df, df_ci=df_ci, **kwargs)


def make_heatmap(data: pd.DataFrame, inverted: bool = False, num_dec: int = 2, skip_zero: bool = True,
                 ax: [None, plt.Axes] = None, df_ci: [pd.DataFrame, None] = None, show_labels: bool = True,
                 show_values: bool = True, value_font_size: int = 4, aspect: float = 0.35,
                 label_font_size: int = 6, x_rotation: int = 0):
    """
    Makes a heatmap.

    Parameters
    ----------
    data: pd.DataFrame
        The data that is used for the values
    inverted: bool
        If True, the values of da. If bool and False, the data values are used for colours, but
        inverted
    num_dec: int or iterable of int
        Number of decimals in which the data is shown. If iterable, one value per row
    skip_zero: bool or iterable of bool
        Whether to skip the "0" in a "0.xx" value
    ax: plt.Axes or None
        If not None, the heatmap will be plot in this Axes. Otherwise a new Axes is created
    df_ci: pd.DataFrame
        DataFrame with confidence interval values
    show_labels: bool
        If True, labels are shown
    show_values: bool
        If True, values are shown
    value_font_size: int
        The fontsize of the text in the cell
    label_font_size: int
        The fontsize of the text on the axislabels
    aspect: float
        The aspect parameter passed to the (created) ax
    x_rotation: int
        Rotation of the x labels.

    #TODO: estimate value/label font + aspect from data/text

    Returns
    -------
    f: plt.Figure
        if ax is None, a new Figure. If ax is not None, this is skipped
    ax: plt.Axes
        if ax is None, a new Axes. Otherwise, ax

    """

    if df_ci is not None:
        assert (df_ci.index == data.index).all()
        assert (df_ci.columns == data.columns).all()

    # Data for colours
    bot = data.min().min()
    top = data.max().max()
    data_for_colours = (data.to_numpy() - bot) / (top - bot)
    if inverted:
        data_for_colours = 1 - data_for_colours

    # Extend skip_zero to rows if necessary
    if isinstance(skip_zero, bool):
        skip_zero = [skip_zero] * len(data)

    # Extend num_dec to rows if necessary
    if isinstance(num_dec, int):
        num_dec = [num_dec] * len(data)

    # Create figure if necessary
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = None

    # Base cmap
    base = cm.get_cmap('inferno', 1000)

    # Heatmap + axis configuration
    ax.imshow(data_for_colours, cmap=base)

    if show_values:
        for x in range(len(data.columns)):
            for y, skip_zero_for_row, num_dec_for_row in zip(range(len(data.index)), skip_zero, num_dec):
                if pd.isna(data.iloc[y, x]):
                    continue

                def fmt(v):
                    s = f'{v:.{num_dec_for_row}f}'
                    if skip_zero_for_row and abs(v) < 1:
                        s = s.replace('0.', '.')
                    return s

                # White in the bottom, black in the top
                col = 'w' if (data_for_colours[y, x] < 0.5) else 'k'

                # Format number of decimals
                t = fmt(data.iloc[y, x])

                # Add confidence interval if necessary
                if df_ci is not None:
                    t = t + rf'$\pm$' + fmt(df_ci.iloc[y, x])

                # Add text to plot
                ax.text(x, y, t, color=col, ha='center', va='center', fontdict=dict(size=value_font_size))

    # Add the aspect
    ax.set_aspect(aspect=aspect)

    # Add the labels
    if show_labels:
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns, fontdict=dict(size=label_font_size), rotation=x_rotation)
        ax.set_yticklabels(data.index, fontdict=dict(size=label_font_size))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Return
    if f is None:
        return ax
    else:
        return f, ax
