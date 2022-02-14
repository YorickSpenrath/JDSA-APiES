import itertools
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from functions import file_functions
from ...data_handlers import AbstractDataHandler, EventLogDataHandler
from ...execution_modules import combine_results
from ...execution_modules.single import run_experiment


def execute(adh_list: Iterable[AbstractDataHandler], fn_out):
    if not isinstance(adh_list, list):
        l1, l2 = itertools.tee(adh_list)
    else:
        l1 = adh_list
        l2 = adh_list

    for adh in l1:
        run_experiment(adh)

    combine_results(l2, fn_out)


def creator(cluster_type, cluster_method):
    clustering_parameter_type = {'k': 'cluster_count', 'rho': 'cluster_size'}[cluster_type]

    def create(model_ki):
        model, ki = model_ki
        return EventLogDataHandler(name=f'{model}_{cluster_method}_{ki}_{cluster_type}',
                                   dataset='bpic2019',
                                   clustering_parameter_type=clustering_parameter_type,
                                   clustering_method=cluster_method,
                                   clustering_parameter=ki,
                                   model_type=model)

    return create


fd_base = Path('results/EventLogDataHandler')
fd_individual = fd_base / 'individual'
fd_res = fd_base / 'common/bpic2019/results'


def combine_all():
    fd_list = file_functions.list_dirs(fd_individual)
    l2 = map(lambda x: EventLogDataHandler(name=x.name), fd_list)
    combine_results(l2, fd_res / 'all.csv')


def experiment_small(cluster_type, cluster_method):
    create = creator(cluster_type, cluster_method)
    small_list = map(create, itertools.product(['AHOT', 'TestModel'], [1, 2, 4, 8, 16, 32, 64, 128, None]))
    execute(small_list, fd_res / f'{cluster_type}_{cluster_method}_small.csv')


def experiment_large(cluster_type, cluster_method):
    create = creator(cluster_type, cluster_method)
    large_list = map(create, itertools.product(['AHOT', 'TestModel'], list(range(1, 179)) + [None]))
    execute(large_list, fd_res / f'{cluster_type}_{cluster_method}_large.csv')


def experiment_all():
    from itertools import chain
    z = []
    for cluster_type in ['k', 'rho']:
        for cluster_method in ['native', 'random']:
            create = creator(cluster_type, cluster_method)
            z.append(map(create, itertools.product(['AHOT', 'TestModel'], list(range(1, 179)) + [None])))
    execute(chain(*z), fd_res / f'all_test.csv')


def create_list(df_with_parameters: pd.DataFrame, function_to_create: callable):
    adh_list = []
    for i, r in df_with_parameters.iterrows():
        adh_list.append(function_to_create(**r.to_dict()))
    return adh_list


if __name__ == '__main__':
    experiment_large('rho', 'native')
    experiment_large('rho', 'random')
