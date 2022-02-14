import logging

import numpy as np
import pandas as pd

from bitbooster import strings as bbs
from bitbooster.utils.objects import discrete_clusterable_from_dataframe, vanilla_clusterable_from_dataframe
from .. import constants as sc
from ..data_handlers import AbstractDataHandler, EventLogDataHandler
from ..data_handlers.abstract_gcs_settings import AbstractGCSSettings
from ..data_handlers.consumer_data_handler import ConsumerDataHandler


def training_cluster(adh: AbstractDataHandler, t: int):
    return __cluster(True, adh, t)


def predicting_cluster(adh: AbstractDataHandler, t: int):
    return __cluster(False, adh, t)


def __determine_k(adh, n_entities):
    # Determine k
    if adh.settings.clustering_parameter_type == AbstractGCSSettings.CLUSTER_SIZE:
        # Convert to appropriate k value
        rho = adh.settings.clustering_parameter

        if rho == 1:
            k = None
        elif rho is None:
            k = 1
        elif isinstance(rho, int) and rho > 1:
            k = (n_entities + rho - 1) // rho
        else:
            raise NotImplementedError
    elif adh.settings.clustering_parameter_type == AbstractGCSSettings.CLUSTER_COUNT:
        k = adh.settings.clustering_parameter
    else:
        raise NotImplementedError(f'Not implemented for {adh.settings.clustering_parameter_type}')

    return k


def __native(adh: AbstractDataHandler, n_entities: int, k: int, data: [pd.DataFrame, np.array]):
    log = logging.getLogger(str(adh))

    # Generate Message
    m = f'Native clustering: {n_entities} -> {k}'
    if adh.settings.clustering_parameter_type == AbstractGCSSettings.CLUSTER_SIZE:
        m += f' ({adh.settings.clustering_parameter} ~= {n_entities / k:.2f})'
    log.info(m)

    # Create clusterable
    # TODO : this should be in the settings
    log.info('Creating clusterable')
    if isinstance(adh, EventLogDataHandler):
        c = vanilla_clusterable_from_dataframe(original_data=data, metric=bbs.GOWER, weighted=False)
    elif isinstance(adh, ConsumerDataHandler):
        c = discrete_clusterable_from_dataframe(original_data=data, n=3, metric=bbs.EUCLIDEAN,
                                                weighted=True)
        if k > c.unique_size:
            k = c.unique_size
            log.info(f'Lowered k to {k} because of BitBooster discretization')
    else:
        raise NotImplementedError(f'Unknown which metric is used for data handler of type {type(adh)}')

    # Get labels
    log.info('Clustering')
    return c.voronoi(k)[1]


def __random(t: int, k: int, data: [pd.DataFrame, np.array], log: logging.Logger):
    log.info('Random clustering')
    rng = np.random.RandomState(seed=t)
    return rng.randint(0, k, len(data))


def __cluster(is_training: bool, adh: AbstractDataHandler, t: int):
    log = logging.getLogger(str(adh))
    if adh.clusters_done(t=t, is_training=is_training):
        log.info('Already done')
        return

    # Read data
    log.info('Loading data')
    data = adh.get_cluster_data(t=t, is_training=is_training)
    n_entities = len(data.drop_duplicates())

    # Case 1: no data
    if data.shape[1] == 0:
        # There are not different datapoints
        log.info('No datapoints')
        labels = 0
    else:
        log.info('Determining k')
        k = __determine_k(adh, n_entities)

        # Determine what to do with this value of k
        if k == 1:
            log.info('Single cluster for all entities')
            labels = 0
        elif k is None:
            log.info('Single cluster per entity')
            labels = range(len(data.drop_duplicates()))
            labelled_data = data.drop_duplicates().assign(label=labels)
            labels = data.reset_index().merge(labelled_data).set_index(data.index.name)['label']
        elif isinstance(k, int) and k > 1:
            k = min(k, n_entities)

            if adh.settings.clustering_method == AbstractGCSSettings.NATIVE:
                labels = __native(adh, n_entities, k, data)
            elif adh.settings.clustering_method == AbstractGCSSettings.RANDOM:
                labels = __random(t, k, data, log)
            else:
                raise NotImplementedError(f'Unknown how to cluster with setting {adh.settings.clustering_method}')

        else:
            raise NotImplementedError(f'Not implemented for k = {k} of type {type(k)}')

    # Export results
    log.info('Exporting clustering results')
    sr = pd.Series(index=data.index, data=labels, name=sc.CLUSTER)
    adh.export_clusters(t=t, is_training=is_training, clusters=sr)
