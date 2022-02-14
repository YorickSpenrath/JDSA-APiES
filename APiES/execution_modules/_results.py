from .. import constants as sc
import logging

import pandas as pd

from functions.dataframe_operations import export_df
from functions.progress import ProgressShower
from APiES.constants import CLUSTER
from ..data_handlers.abstract_data_handler import AbstractDataHandler
from ..data_handlers.event_log_data_handler import EventLogDataHandler


def prepended_index(idx, pre):
    return list(map(lambda x: f'{pre}_{x}', idx))


def compute_results(adh: AbstractDataHandler):
    # We compute metrics in two ways: once at every timestep, and once over all predictions. Each of these is then also
    # split between predictions for clusters and predictions for entities.

    # These dataframes collect every prediction ever made (all-time predictions)
    cluster_df = pd.DataFrame()
    entity_df = pd.DataFrame()

    # These dataframes collect the metrics at every point in time
    cluster_results_over_time = pd.DataFrame()
    entity_results_over_time = pd.DataFrame()

    log = logging.getLogger(str(adh))
    log.info('Extracting predictions')

    # This tracks the previous ground truth. This is required as some DataHandlers (such as the consumer data handler)
    # need the ground truth of the previous time step
    y_prev = {True: pd.Series(dtype=float, name='y_prev'), False: pd.Series(dtype=float, name='y_prev')}
    for t in ProgressShower(adh.timestamps, pre=adh.name):

        # This loop first does the result for clusters, then for entities
        for parse_cluster in [True, False]:

            # Fetch the right values for predictions/ground truth
            if parse_cluster:
                predictions = adh.get_cluster_predictions(t)
                ground_truth = adh.get_cluster_ground_truth(t)
            else:
                predictions = adh.get_entity_predictions(t)
                ground_truth = adh.get_entity_ground_truth(t)

            # Rename the series
            predictions.name = 'y_pred'
            ground_truth.name = 'y_true'

            # Combine the predictions with the ground truth and the previous ground truth
            # doing this in a dataframe is easiest because you can easily remove the non-predicted items, and then
            # append the whole thing to the all-time predictions
            df = predictions \
                .to_frame() \
                .join(ground_truth, how='outer') \
                .join(y_prev[parse_cluster], how='outer') \
                .assign(t=t)

            # Remove all rows (clusters/entities) for which no prediction was made
            df = df[~(df['y_pred'].isna())]

            if parse_cluster:
                # Add to all-time cluster predictions
                cluster_df = cluster_df.append(df.reset_index())

                # Compute metrics for this time step

                # y_prev is None because this value is not computable for clusters
                # TODO, actually, it is, but the 'prev' of a cluster is the previous ground truth
                cluster_results_over_time[t] = adh.compute_results_df(y_true=df['y_true'],
                                                                      y_pred=df['y_pred'],
                                                                      is_cluster=parse_cluster,
                                                                      y_prev=None)
            else:
                # Add to all-time entity predictions
                entity_df = entity_df.append(df.reset_index())

                # Compute metrics for this timestep
                entity_results_over_time[t] = adh.compute_results_df(y_true=df['y_true'],
                                                                     y_pred=df['y_pred'],
                                                                     is_cluster=parse_cluster,
                                                                     y_prev=df['y_prev'])

            # Set the current ground truth as the previous ground truth
            y_prev[parse_cluster] = ground_truth.rename('y_prev')

    # Global results ===================================================================================================
    # These are the results computed over all predictions ever made for a cluster (True) or an entity (False)
    res = dict()
    res_ot = dict()
    log.info('Computing Statistics')
    for parse_cluster in [True, False]:
        if parse_cluster:
            df = cluster_df
            results_over_time = cluster_results_over_time
            pre = CLUSTER
        else:
            df = entity_df
            results_over_time = entity_results_over_time
            pre = adh.datapoint_name

        # y_prev is None since the results in this part are not time-dependent
        results = adh.compute_results_df(y_true=df['y_true'],
                                         y_pred=df['y_pred'],
                                         is_cluster=parse_cluster,
                                         y_prev=None)

        # modify the metric name with cluster/entity name
        def modify(dfx):
            dfx.index = prepended_index(dfx.index, pre)
            dfx.name = sc.VALUE
            dfx.index.name = sc.METRIC
            return dfx

        res[parse_cluster] = modify(results)
        res_ot[parse_cluster] = modify(results_over_time)

    log.info('Exporting results')
    results = res[True].append(res[False])
    export_df(results, adh.fn_results)

    results_over_time = res_ot[True].append(res_ot[False])
    export_df(results_over_time, adh.fn_results_over_time, index=True)


if __name__ == '__main__':
    for i in list(range(1, 128)) + [None]:
        print(i)
        compute_results(EventLogDataHandler(name=f'AHOT_{i}', dataset='bpic2019', k=i, model_type='AHOT'))
        compute_results(EventLogDataHandler(name=f'TestModel_{i}', dataset='bpic2019', k=i, model_type='TestModel'))
