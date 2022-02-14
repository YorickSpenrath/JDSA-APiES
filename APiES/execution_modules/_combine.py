from pathlib import Path
from typing import Iterable

import pandas as pd

from functions import confidence_interval
from functions.dataframe_operations import import_sr, export_df, import_df
from .. import constants as sc
from ..data_handlers import AbstractDataHandler


def combine_results(list_of_adh: Iterable[AbstractDataHandler], fn_out: [str, Path]):
    """
    Combines the results of multiple experiments

    Parameters
    ----------
    list_of_adh: Iterable of AbstractDataHandler or str or Path
        List of AbstractDataHandlers describing the experiments. If str of Path, all folders in the path are considered
    fn_out: str or Path
        The save location of the combined results

    """

    # All results
    df_results = pd.DataFrame()

    # Iterate over all ADHs
    for adh in list_of_adh:

        if not adh.fn_results.exists():
            continue

        # Add results from the single adh file
        df_results[adh.name] = import_sr(adh.fn_results)

        # Add each parameter from the settings
        for k, v in adh.settings.as_dict().items():
            df_results.loc[k, adh.name] = v

    # Transpose and export
    df_results = df_results.T
    df_results.index.name = sc.EXPERIMENT
    export_df(df_results, fn_out, index=True)


def combine_time_results(list_of_adh: Iterable[AbstractDataHandler],
                         fn_mean: [str, Path],
                         fn_ci: [str, Path],
                         ci: [float] = .95):
    """
    Extracts the results over time for all experiments and writes the mean and std to given files

    Parameters
    ----------
    list_of_adh: Iterable of AbstractDataHandler
        List of AbstractDataHandlers for which to get the results. ADH's that are not finished are skipped
    fn_mean: str or Path
        Location to save the mean values
    fn_ci: str or Path
        Locations to save the ci values
    ci: float
        Confidence level, in [0,1]
    """
    df_mean = pd.DataFrame()
    df_ci = pd.DataFrame()

    for adh in list_of_adh:
        if not adh.fn_results_over_time.exists():
            continue

        df = import_df(adh.fn_results_over_time).set_index(sc.METRIC)

        df_mean[adh.name] = df.apply('mean', axis=1)
        df_ci[adh.name] = df.apply(confidence_interval.ci_gb_function(ci), axis=1)

        for k, v in adh.settings.as_dict().items():
            df_mean.loc[k, adh.name] = v
            df_ci.loc[k, adh.name] = v

    for (df, fn) in zip([df_mean, df_ci], [fn_mean, fn_ci]):
        df = df.T
        df.index.name = sc.EXPERIMENT
        export_df(df, fn, index=True)
