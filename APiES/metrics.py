import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, balanced_accuracy_score, \
    recall_score, precision_score, f1_score


def ada_r2(y_true, y_pred):
    if len(y_true) == 1:
        return pd.NA
    else:
        return r2_score(y_true, y_pred)


def rmse_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def amape(y_true, y_pred):
    y_pred = y_pred[y_true != 0]
    y_true = y_true[y_true != 0]
    return mean_absolute_percentage_error(y_true, y_pred)


# TODO make this part faster
def decile_x(f, _is_bottom):
    def my_fun(y_true, y_pred):
        y_true = k_fraction_label(y_true, _is_bottom)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return pd.NA
        y_pred = k_fraction_label(y_pred, _is_bottom)

        assert isinstance(y_pred, pd.Series)
        assert isinstance(y_true, pd.Series)
        return f(y_true, y_pred)

    return my_fun


top_k_fraction = 0.1


def k_fraction_label(sr_, is_bottom, tkf=top_k_fraction):
    srx_ = sr_.sort_index()
    srx_ = srx_.sort_values(kind='mergesort', ascending=is_bottom)
    srx_.iloc[:int(tkf * len(sr_))] = True
    srx_.iloc[int(tkf * len(sr_)):] = False
    return srx_.astype(bool).sort_index()


def classification_thing(df_turnover):
    df_turnover = df_turnover.replace({0: pd.NA})

    a = ~df_turnover.prev.isna()
    b = ~df_turnover.this.isna()
    c = ~df_turnover.pred.isna()

    abs_true_labels = pd.Series(index=df_turnover.index, dtype=float)
    rel_true_labels = pd.Series(index=df_turnover.index, dtype=float)
    abs_pred_labels = pd.Series(index=df_turnover.index, dtype=float)
    rel_pred_labels = pd.Series(index=df_turnover.index, dtype=float)

    def get_abs_true(mask):
        return df_turnover.prev[mask] - df_turnover.this[mask]

    def get_abs_pred(mask):
        return df_turnover.prev[mask] - df_turnover.pred[mask]

    m = a & b & c
    # All good
    abs_true_labels[m] = get_abs_true(m)
    abs_pred_labels[m] = get_abs_pred(m)
    rel_true_labels[m] = get_abs_true(m).divide(df_turnover.prev[m])
    rel_pred_labels[m] = get_abs_pred(m).divide(df_turnover.prev[m])

    m = ~a & b & c
    # No previous purchase
    abs_true_labels[m] = -df_turnover.this[m]
    abs_pred_labels[m] = -df_turnover.pred[m]
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    m = a & ~b & c
    # No purchase in last week
    abs_true_labels[m] = df_turnover.prev[m]
    abs_pred_labels[m] = get_abs_pred(m)
    rel_true_labels[m] = 1
    rel_pred_labels[m] = get_abs_pred(m).divide(df_turnover.prev[m])

    m = a & b & ~c
    # Previous but no predicted purchase, this should not be possible
    abs_true_labels[m] = get_abs_true(m)
    abs_pred_labels[m] = df_turnover.prev[m]
    rel_true_labels[m] = get_abs_true(m).divide(df_turnover.prev[m])
    rel_pred_labels[m] = 1

    m = a & ~b & ~c
    # Previous but no predicted purchase, this should not be possible
    abs_true_labels[m] = df_turnover.prev[m]
    abs_pred_labels[m] = df_turnover.prev[m]
    rel_true_labels[m] = 1
    rel_pred_labels[m] = 1

    m = ~a & b & ~c
    # New purchase, not predicted. These consumers are not important
    abs_true_labels[m] = -np.inf
    abs_pred_labels[m] = -np.inf
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    m = ~a & ~b & c
    # Prediction, but it is not relevant
    abs_true_labels[m] = -np.inf
    abs_pred_labels[m] = -np.inf
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    m = ~a & ~b & ~c
    # These consumers never purchased anything, and were not predicted to do so
    abs_true_labels[m] = -np.inf
    abs_pred_labels[m] = -np.inf
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    # Verify we have everything
    for sr in [abs_true_labels, abs_pred_labels, rel_true_labels, rel_pred_labels]:
        assert sr.isna().sum() == 0

    abs_true_labels = top_k_fraction_label(abs_true_labels)
    abs_pred_labels = top_k_fraction_label(abs_pred_labels)
    rel_true_labels = top_k_fraction_label(rel_true_labels)
    rel_pred_labels = top_k_fraction_label(rel_pred_labels)

    metric_names = ['accuracy', 'recall', 'precision', 'f1']
    metric_methods = [balanced_accuracy_score, recall_score, precision_score, f1_score]

    res = pd.Series(dtype=float)

    for mn, m in zip(metric_names, metric_methods):

        # Absolute
        if abs_true_labels.all() or not abs_true_labels.any():
            x = pd.NA
        elif abs_pred_labels.all() or not abs_pred_labels.any():
            x = pd.NA
        else:
            x = m(abs_true_labels, abs_pred_labels)
        res[f'bottom_absolute_{mn}'] = x

        # Relative
        if rel_true_labels.all() or not rel_true_labels.any():
            x = pd.NA
            res[f'bottom_relative_{mn}'] = pd.NA
        elif rel_pred_labels.all() or not rel_pred_labels.any():
            x = pd.NA
        else:
            x = m(rel_true_labels, rel_pred_labels)

        res[f'bottom_relative_{mn}'] = x

    return res


def top_k_fraction_label(sr_, tkf=top_k_fraction):
    return k_fraction_label(sr_, is_bottom=False, tkf=tkf)
