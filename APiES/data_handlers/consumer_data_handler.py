import functools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from functions import file_functions
from functions.dataframe_operations import import_sr, import_df, export_df
from .consumer_settings import ConsumerCSSettings
from .. import constants as sc
from ..data_handlers import AbstractDataHandler
from APiES.metrics import classification_thing

from APiES.constants import CCS_ROOT

try:
    from generalized_ccs._cdh_local import time_name_ty, cdl_root, fn_birth_ty
except (ModuleNotFoundError, ImportError):

    cdl_root = Path('data') / 'shopper_data'
    fn_birth_ty = Path('data') / 'shopper_births'
    if cdl_root.exists():
        time_name_ty = {i: fn.name[:-4] for i, fn in sorted(enumerate(file_functions.list_files(cdl_root)))}
    else:
        time_name_ty = dict()

# This is a constant for now
tb = 1


class ConsumerDataHandler(AbstractDataHandler):

    # Settings ---------------------------------------------------------------------------------------------------------
    @property
    def settings_class(self):
        return ConsumerCSSettings

    @property
    def settings(self) -> ConsumerCSSettings:
        return ConsumerCSSettings(self.fn_settings)

    # Feature / target -------------------------------------------------------------------------------------------------
    def get_training_cluster_xy(self, t: int) -> (pd.DataFrame, pd.Series):
        is_training = True
        clusters = self._get_clusters(t, is_training)
        x = get_x(t0=t - self.settings.tau - tb * is_training, te=t - tb * is_training, clusters=clusters)
        y = get_y(t0=t - tb * is_training, te=t + tb * (not is_training), clusters=clusters)
        return x, y

    def get_predicting_cluster_x_and_index(self, t: int) -> (np.array, pd.Index):
        is_training = False
        clusters = self._get_clusters(t, is_training)
        x = get_x(t0=t - self.settings.tau - tb * is_training, te=t - tb * is_training, clusters=clusters)
        index = clusters.sort_values().drop_duplicates().to_numpy()
        return x, index

    # Ground Truth -----------------------------------------------------------------------------------------------------
    @property
    def fd_ground_truth(self):
        return self.fd_common / 'ground_truths'

    @property
    def fn_all_cluster_ground_truths(self):
        assert tb == 1, 'Not implemented for tb!=1'
        return self.fd_ground_truth \
               / f'cm={self.settings.clustering_method}' \
               / f'tau={self.settings.tau}' \
               / f'{self.settings.param}={self.settings.clustering_parameter}.csv'

    @property
    def fn_all_turnover_ground_truths(self):
        return self.fd_ground_truth / f'turnover.csv'

    def get_cluster_ground_truth(self, t: int) -> pd.Series:
        fn = self.fn_all_cluster_ground_truths
        if not fn.exists():
            combine_ground_truths(self)
        return import_df(fn).set_index(sc.CLUSTER)[str(t)]

    @property
    def fn_all_consumer_ground_truths(self) -> Path:
        assert tb == 1, 'Not implemented for tb!=1'
        return self.fd_ground_truth / f'consumer_ty.csv'

    def get_entity_ground_truth(self, t: int) -> pd.Series:
        fn = self.fn_all_consumer_ground_truths
        if not fn.exists():
            combine_ground_truths(self)
        df = import_df(fn).set_index(sc.consumer)
        df.columns = map(int, df.columns)
        if t in df.columns:
            return df[t]
        else:
            return pd.Series(dtype=float)

    # Clusters ---------------------------------------------------------------------------------------------------------
    @property
    def fd_clusters(self) -> Path:
        return self.fd_common \
               / f'clusters' \
               / f'cm={self.settings.clustering_method}' \
               / f'tau={self.settings.tau}' \
               / f'{self.settings.param}={self.settings.clustering_parameter}'

    def fn_clusters(self, t: int, is_training: bool) -> Path:
        return self.fd_clusters / f'{t - is_training}.csv'

    def _export_clusters(self, t: int, is_training: bool, clusters: pd.Series) -> None:
        export_df(clusters, fn=self.fn_clusters(t, is_training))

    def clusters_done(self, t: int, is_training: bool) -> bool:
        return self.fn_clusters(t, is_training).exists()

    def get_cluster_data(self, t: int, is_training: bool) -> pd.DataFrame:
        # Note: is_training is not used, as the clusters are the same for training/testing

        if is_training:
            t0 = t - tb - self.settings.tau
            t1 = t - tb
        else:
            t0 = t - self.settings.tau
            t1 = t

        # Timeframe for the cluster data
        # Why is there a '<' on the right bound? We are dealing with data that is stored per week. As such, the interval
        # from timestamp t0 to timestamp t1 does not involve the data stored in the file {t1}.csv, but does at
        # {t1-1}.csv
        all_times = [t_prime for t_prime in time_name_ty.keys() if t0 <= t_prime < t1]

        # TODO there are some consumers that do not have any purchases in train_times/all_times
        # Consumers that started at or before the timeframe
        sr_births = import_sr(fn_birth_ty)
        eligible_consumers = sr_births[sr_births <= t - self.settings.tau - tb].index

        # Import data
        def imp(bn):
            return import_df(consumer_description_log(bn)).set_index(sc.consumer)

        f_matrices = [imp(time_name_ty[k]) for k in all_times]

        # Correct for consumers not having shopped in certain weeks
        all_consumers = set().union(*[set(df.index) for df in f_matrices]).intersection(eligible_consumers)
        all_consumers = pd.Index(all_consumers, name=self.datapoint_name)
        # TODO fill_value=0 is not completely right for some features...
        f_matrices = [df.reindex(all_consumers, fill_value=0) for df in f_matrices]

        # All features that are considered
        features = sorted([k for k in sc.keep_feats if k != sc.NUMBER_OF_WEEKS])

        # Only add residuals if there are more than 2 weeks
        add_r = len(f_matrices) > 2

        # Data holder: n_consumers * (n_polyfit_parameters * n_features)
        data = np.empty((len(all_consumers), (2 + add_r) * len(features)), dtype=float)

        # Get the sequence of values for a given feature
        def mat(feature):
            return np.array([df.loc[:, feature].to_numpy() for df in f_matrices])

        # Compute the polyfit parameters for each feature
        for i, f in enumerate(features):
            # Compute polyfit values
            (a, b), r, *_ = np.polyfit(np.array(all_times), mat(f), 1, full=True)

            # Store values in correct place
            data[:, (2 + add_r) * i + 0] = a
            data[:, (2 + add_r) * i + 1] = b
            if add_r:
                data[:, (2 + add_r) * i + 2] = r

        # Adapted feature names
        adapted = sum([[f + '_' + i for i in list('abr' if add_r else 'ab')] for f in features], [])

        # Polyfit parameters converted to dataframe
        return pd.DataFrame(data=data, columns=adapted, index=all_consumers)

    def _get_clusters(self, t: int, is_training: bool) -> pd.Series:
        return import_sr(self.fn_clusters(t, is_training)).astype(int)

    # Properties -------------------------------------------------------------------------------------------------------
    @property
    def datapoint_name(self) -> str:
        return sc.consumer

    @property
    def timestamps(self) -> Iterable[int]:
        return [ti for ti in time_name_ty.keys() if ti >= self.settings.tau + tb][:-1]

    @property
    def fd_base(self):
        return CCS_ROOT

    @staticmethod
    def compute_results_df(y_true: pd.Series, y_pred: pd.Series, is_cluster: bool, y_prev: [pd.Series, None] = None):
        """
        Parameters
        ----------
        y_true: pd.Series
            Series with ground truth
        y_pred: pd.Series
            Series with predicted values
        is_cluster: bool
            Whether the given data belongs to clusters (True) or entities (False)
        y_prev: pd.Series or None
            Series with ground truth of previous timestamp

        Returns
        -------
        res: pd.Series
            Series with all metrics that are relevant for this Data Handler

        """
        # Fix thanks to: https: // stackoverflow.com / a / 26807879 / 14781275
        y_true = y_true.fillna(0)
        res_sr = super(ConsumerDataHandler, ConsumerDataHandler).compute_results_df(y_true, y_pred, is_cluster, y_prev)

        true_turnover = y_true.sum()
        pred_turnover = y_pred.sum()
        res_sr['turnover'] = abs(true_turnover - pred_turnover) / true_turnover

        if y_prev is not None:
            df = pd.DataFrame({'this': y_true, 'prev': y_prev, 'pred': y_pred})
            turnover_drop_sr = classification_thing(df)

            def convert(x):
                return x.replace('bottom_', 'turnover_drop_')

            turnover_drop_sr.index = map(convert, turnover_drop_sr.index)
            res_sr = res_sr.append(turnover_drop_sr)

        return res_sr


def consumer_description_log(base_name):
    return cdl_root / f'{base_name}.csv'


def get_dfs_from_range(t0, te):
    def imp(bn):
        return import_df(consumer_description_log(bn)).set_index(sc.consumer)

    return [imp(v) for k, v in time_name_ty.items() if t0 <= k < te]


def get_y(t0, te, clusters):
    assert te - t0 == 1, 'This is wrong for te-t0!=1'
    test_data = get_dfs_from_range(t0, te)
    n_pictures = len(clusters.unique())
    y = np.empty(shape=(n_pictures,), dtype=np.float)
    for cluster, dfx in clusters.to_frame().reset_index().groupby(sc.CLUSTER, sort=True):
        y[cluster] = sum([df.reindex(dfx[sc.consumer], fill_value=0)[sc.TOTAL_VALUE].mean() for df in test_data])

    return y


def get_x(t0, te, clusters):
    train_data = get_dfs_from_range(t0, te)

    n_pictures = len(clusters.unique())
    picture_height = len(train_data)
    picture_width = len(sc.keep_feats)

    x = np.empty(shape=(n_pictures, picture_height, picture_width), dtype=np.float)

    for cluster, dfx in clusters.to_frame().reset_index().groupby(sc.CLUSTER, sort=True):
        for i, td in enumerate(train_data):
            x[cluster, i, :] = td.reindex(dfx[sc.consumer], fill_value=0)[sc.keep_feats].mean(axis=0)

    return x


def get_consumer_turnover(t0, te):
    def red(x, y):
        return x[sc.TOTAL_VALUE].add(y[sc.TOTAL_VALUE], fill_value=0)

    z = get_dfs_from_range(t0=t0, te=te)

    return functools.reduce(red, z, pd.DataFrame(columns=[sc.TOTAL_VALUE]))


def combine_ground_truths(cdh: ConsumerDataHandler):
    assert tb == 1, 'Not implemented for tb != 1'
    fn_consumer = cdh.fn_all_consumer_ground_truths
    fn_cluster = cdh.fn_all_cluster_ground_truths
    if all(map(lambda x: x.exists(), [fn_consumer, fn_cluster, cdh.fn_all_turnover_ground_truths])):
        return

    if fn_consumer.exists():
        df_consumer = import_df(fn_consumer).set_index(sc.consumer)
        df_consumer.columns = map(int, df_consumer.columns)
    else:
        df_consumer = pd.DataFrame()

    if fn_cluster.exists():
        df_cluster = import_df(fn_cluster).set_index(sc.CLUSTER)
        df_cluster.columns = map(int, df_cluster.columns)
    else:
        df_cluster = pd.DataFrame()

    for t in cdh.timestamps:
        if t not in df_consumer.columns:
            sr_t = get_consumer_turnover(t0=t, te=t + tb)
            sr_t.name = t
            df_consumer = df_consumer.join(other=sr_t, how='outer').fillna(0)
        if t not in df_cluster.columns:
            sr_t = get_y(t0=t, te=t + tb, clusters=import_sr(cdh.fn_clusters(t, is_training=False)))
            sr_t = pd.Series(sr_t, name=t)
            df_cluster = df_cluster.join(other=sr_t, how='outer').fillna(0)

    df_consumer.index.name = sc.consumer
    df_cluster.index.name = sc.CLUSTER

    export_df(df_consumer[sorted(df_consumer.columns)], fn_consumer, index=True)
    sr_turnover = df_consumer.sum(axis=0)
    sr_turnover.index.name = 't'
    sr_turnover.name = 'turnover'
    export_df(sr_turnover, cdh.fn_all_turnover_ground_truths, index=True)
    export_df(df_cluster[sorted(df_cluster.columns)], fn_cluster, index=True)
