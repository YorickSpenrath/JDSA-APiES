import numpy as np
from pathlib import Path

import abc
from typing import Iterable

import pandas as pd
from sklearn.metrics import precision_score, recall_score, mean_absolute_percentage_error

from functions import file_functions
from functions.alm import AbstractLocationsManager
from functions.dataframe_operations import export_df, import_sr, import_df
from .. import constants as sc
from ..data_handlers.abstract_gcs_settings import AbstractGCSSettings
from APiES.timed_models import TestModel, AHOT, AbstractTimedModel, TaxTimedModel


class AbstractDataHandler(AbstractLocationsManager):

    def __init__(self, name, **kwargs):
        """
        Creates a DataHandler

        Parameters
        ----------
        name: str
            Name for the experiment

        Other Parameters
        ----------------
        Settings for AbstractGCSSettings
        """
        super().__init__(name=name, **kwargs)
        if self.settings.model_type == 'TestModel':
            self._model = TestModel()
        elif self.settings.model_type == 'AHOT':
            self._model = AHOT(self.fd_models)
        elif self.settings.model_type == 'TAX':
            self._model = TaxTimedModel(self.fd_models)
        else:
            raise NotImplementedError('ModelType')

    @property
    def fd_models(self):
        return self.fd / 'models'

    def __repr__(self):
        """
        Name of the class
        Returns
        -------
        repr: str
            Class_name[experiment_name]
        """
        return f'{self.__class__.__name__}[{self.name}]'

    @property
    def fn_results(self):
        """
        Place where the results are saved

        Returns
        -------
        fn: Path
            Location for the results
        """
        return self.fd / 'results.csv'

    @property
    def fn_results_over_time(self):
        """
        Place where the results over time are saved

        Returns
        -------
        fn: Path
            Location for the results over time
        """
        return self.fd / 'results_over_time.csv'

    @property
    def model(self) -> AbstractTimedModel:
        """
        Model used for the experiment

        Returns
        -------
        model: AbstractTimedModel
            Timed model used in the experiments
        """
        return self._model

    @property
    def settings(self) -> AbstractGCSSettings:
        """
        Settings for this experiment

        Returns
        -------
        settings: AbstractGCSSettings
            Settings for this experiment, a subclass of AbstractGCSSettings
        """
        return self._settings

    # XY data ==========================================================================================================
    @abc.abstractmethod
    def get_training_cluster_xy(self, t: int) -> (pd.DataFrame, pd.Series):
        """
        Get the x and y data for the given time t for training

        Parameters
        ----------
        t: int
            The time for which to get the x/y

        Returns
        -------
        x: np.ndarray of size (n_clusters, n_features)
            x data (cluster average) at time t for training
        y: np.array of size (n_clusters,)
            y data (cluster average) at time t for training
        """
        pass

    @abc.abstractmethod
    def get_predicting_cluster_x_and_index(self, t: int) -> (np.array, pd.Index):
        pass

    @abc.abstractmethod
    def get_cluster_ground_truth(self, t: int):
        pass

    # Clusters Done ====================================================================================================
    @abc.abstractmethod
    def clusters_done(self, t: int, is_training: bool) -> bool:
        pass

    # Clusters =========================================================================================================
    @abc.abstractmethod
    def _get_clusters(self, t: int, is_training: bool) -> pd.Series:
        pass

    def get_training_clusters(self, t: int) -> pd.Series:
        return self._get_clusters(t, is_training=True)

    def get_predicting_clusters(self, t: int) -> pd.Series:
        return self._get_clusters(t, is_training=False)

    # Save Clusters ====================================================================================================
    @abc.abstractmethod
    def _export_clusters(self, t: int, is_training: bool, clusters: pd.Series) -> None:
        pass

    def export_clusters(self, t: int, is_training: bool, clusters: pd.Series) -> None:
        assert clusters.index.name == self.datapoint_name
        assert clusters.name == sc.CLUSTER
        self._export_clusters(t, is_training, clusters)

    # Cluster Data =====================================================================================================
    @abc.abstractmethod
    def get_cluster_data(self, t: int, is_training: bool) -> pd.DataFrame:
        pass

    # Cluster Predictions ==============================================================================================
    @property
    def fd_cluster_predictions(self) -> Path:
        return self.fd / 'cluster_predictions'

    def fn_cluster_predictions(self, t: int) -> Path:
        return self.fd_cluster_predictions / f'{t}.csv'

    def export_cluster_predictions(self, t: int, predictions: pd.Series) -> None:
        assert predictions.index.name == sc.CLUSTER
        assert predictions.name == sc.CLUSTER_PREDICTION
        export_df(predictions, self.fn_cluster_predictions(t))

    def get_cluster_predictions(self, t: int) -> pd.Series:
        if self.fn_all_cluster_predictions.exists():
            # cleaned up
            return import_df(self.fn_all_cluster_predictions) \
                .set_index(sc.CLUSTER)[str(t)] \
                .rename(sc.CLUSTER_PREDICTION)
        else:
            return import_sr(self.fn_cluster_predictions(t))

    def cluster_predictions_done(self, t: int) -> bool:
        return self.fn_cluster_predictions(t).exists()

    # Entity Predictions ===============================================================================================
    def get_entity_predictions(self, t: int) -> pd.Series:
        clusters = self.get_predicting_clusters(t)
        cluster_predictions = self.get_cluster_predictions(t)
        return clusters \
            .to_frame() \
            .merge(right=cluster_predictions, how='left', left_on=sc.CLUSTER, right_index=True) \
            .rename(columns={sc.CLUSTER_PREDICTION: sc.ENTITY_PREDICTION})[sc.ENTITY_PREDICTION] \
            .dropna()

    @abc.abstractmethod
    def get_entity_ground_truth(self, t: int) -> pd.Series:
        pass

    # Timestamps =======================================================================================================
    @property
    @abc.abstractmethod
    def timestamps(self) -> Iterable[int]:
        """
        The timestamps that exist in this dataset

        Returns
        -------
        timestamps: Iterable[int]
            The timestamps
        """
        pass

    @property
    @abc.abstractmethod
    def datapoint_name(self) -> str:
        pass

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
        # TODO child classes now implement checks/edits, which should be done as explicit separate method
        from ..metrics import rmse_score, ada_r2, amape, decile_x
        from sklearn.metrics import f1_score, balanced_accuracy_score

        assert (y_true.index == y_pred.index).all()

        # TODO just return an Nan-series if len(y_true) is 0
        def empty_wrap(fun):
            def my_fun(yt, yp):
                if len(yt) == 0:
                    return pd.NA
                else:
                    return fun(yt, yp)

            return my_fun

        def len_x(yt, yp):
            assert len(yt) == len(yp)
            return len(yt)

        # metric_name, metric_function, metric_for_clusters
        # TODO: decile things don't mean anything for clusters? But just computing them allows for cleaner code
        metrics = [
            # Cluster and Entity functions
            ('rmse', empty_wrap(rmse_score), True),
            ('count', len_x, True),
            ('r2', empty_wrap(ada_r2), True),
            ('mape', empty_wrap(mean_absolute_percentage_error), True),
            ('mape0', empty_wrap(amape), True),
        ]

        for name, is_bottom in zip(['top', 'bottom'], [False, True]):
            metrics += [
                (f'{name}_decile_accuracy', empty_wrap(decile_x(balanced_accuracy_score, is_bottom)), False),
                (f'{name}_decile_f1', empty_wrap(decile_x(f1_score, is_bottom)), False),
                (f'{name}_decile_precision', empty_wrap(decile_x(precision_score, is_bottom)), False),
                (f'{name}_decile_recall', empty_wrap(decile_x(recall_score, is_bottom)), False),
            ]

        res_sr = pd.Series(dtype=float, name='value')
        for metric_name, metric_method, cluster_metric in metrics:
            if cluster_metric or not is_cluster:
                res_sr[metric_name] = metric_method(y_true, y_pred)
        res_sr.index.name = 'metric'
        return res_sr

    # TODO put this in the alm instead of here
    @property
    def fd(self):
        return self.fd_base / 'individual' / self.name

    @property
    def fn_all_cluster_predictions(self):
        return self.fd / 'all_cluster_predictions.csv'

    def cleanup(self):
        # Combine all cluster predictions
        if not self.fn_all_cluster_predictions.exists():
            z = pd.DataFrame()
            for t in self.timestamps:
                z[t] = self.get_cluster_predictions(t)
            export_df(z, self.fn_all_cluster_predictions, index=True)
        file_functions.delete(self.fd_cluster_predictions)

    @property
    def log_file(self):
        return self.fd / 'log.csv'

    def is_done(self):
        return self.fn_results.exists() and self.fn_results_over_time.exists()
