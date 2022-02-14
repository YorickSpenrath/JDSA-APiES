from typing import Iterable

from functions import file_functions
from APiES.constants import CCS_ROOT
from functions.dataframe_operations import import_df
from ...data_handlers import ConsumerDataHandler
from ...execution_modules import combine_results, combine_time_results


def combine(param):
    """
    Creates files combining the results of all experiments, measured as global, mean and ci, of the given param

    Parameters
    ----------
    param: str
        'k' or 'rho', indicating which experiments

    """
    # Combine global results
    adh_list = all_completed_cdh(param)
    combine_results(adh_list, fn_combined_results_global(param))

    # Combine over-time results
    adh_list = all_completed_cdh(param)
    fn_mean = fn_combined_results_time_mean(param)
    fn_ci = fn_combined_results_time_ci(param)
    combine_time_results(adh_list, fn_mean, fn_ci)


def fn_combined_results_global(param):
    """
    File containing the combined results of all experiments, measured as global values, for given param

    Parameters
    ----------
    param: str
        'k' or 'rho', indicating which experiments

    Returns
    -------
    fn: Path
        Location of the of the results for `param'

    """

    return CCS_ROOT / 'common' / f'{param}_results.csv'


def fn_combined_results_time_mean(param):
    """
    File containing the combined results of all experiments, measured as mean over time, for given param

    Parameters
    ----------
    param: str
        'k' or 'rho', indicating which experiments

    Returns
    -------
    fn: Path
        Location of the mean of the results over time for `param'

    """
    return CCS_ROOT / 'common' / f'{param}_time_mean.csv'


def fn_combined_results_time_ci(param):
    """
    File containing the combined results of all experiments, measured as CI over time, for given param

    Parameters
    ----------
    param: str
        'k' or 'rho', indicating which experiments

    Returns
    -------
    fn: Path
        Location of the ci of the results over time for `param'

    """
    return CCS_ROOT / 'common' / f'{param}_time_ci.csv'


def all_completed_cdh(param: [str, None]) -> Iterable[ConsumerDataHandler]:
    """
    Fetches all completed experiments for the given 'rho'/'k'

    Parameters
    ----------
    param: str or None
        'k' or 'rho', indicating which experiments. If None, both are returned

    Returns
    -------
    cdh_list: Iterable[ConsumerDataHandler]
        Collection of ConsumerDataHandlers that are of the correct experiment type, and are finished.

    """
    if not (CCS_ROOT / 'individual').exists():
        return []
    if param == 'rho':
        ret = map(ConsumerDataHandler,
                  filter(lambda x: '_rho=' in x, file_functions.list_dirs(CCS_ROOT / 'individual', False)))
    elif param == 'k':
        ret = map(ConsumerDataHandler,
                  filter(lambda x: '_k=' in x, file_functions.list_dirs(CCS_ROOT / 'individual', False)))
    elif param is None:
        ret = map(ConsumerDataHandler, file_functions.list_dirs(CCS_ROOT / 'individual', False))
    else:
        raise NotImplementedError(param)

    return filter(lambda x: x.is_done(), ret)
