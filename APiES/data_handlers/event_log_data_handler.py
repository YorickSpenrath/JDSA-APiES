from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from functions import file_functions
from .. import constants as sc
from APiES.constants import EVENT_LOG_ROOT
from ..data_handlers.abstract_data_handler import AbstractDataHandler
from ..data_handlers.event_log_gcs_settings import EventLogCSSettings


class EventLogDataHandler(AbstractDataHandler):
    @property
    def datapoint_name(self) -> str:
        return sc.CASE

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # Todo: based on
        #  self.model.needs_numerical
        #  self.model.needs_sequence
        #  determine what the actual input should look like (sequence/numeric/regular data)

        self._activity_frequencies = None
        self._times = None
        self._clusters = None
        self._predictions = None
        self._case_features_non_ohe = None
        self._case_features_ohe = None

    def prepare_aft(self):
        """
        Prepares the Activity Frequency and Times data
        """
        # Load Event Log ===============================================================================================
        # Load
        event_log = pd.read_csv(self.fn_events)
        # Check all features are present
        assert set(event_log.columns) == {sc.CASE, sc.ACTIVITY, sc.TIMESTAMP}
        # Fix the timestamps
        event_log[sc.TIMESTAMP] = pd.to_datetime(event_log[sc.TIMESTAMP])

        # Flag and check Prediction/Target activities ==================================================================
        for (activity_type, activity_name) in zip(['prediction', 'target'],
                                                  [self.settings.prediction_activity, self.settings.target_activity]):
            # Set flag
            event_log[f'is_{activity_type}_activity'] = (event_log[sc.ACTIVITY] == activity_name)

            # Check flag
            assert (event_log.groupby(sc.CASE)[f'is_{activity_type}_activity'].sum() == 1).all(), \
                f'Some cases have multiple {activity_type} activities'

        # Prepare times ================================================================================================
        if self.fn_times.exists():
            # Load time data
            self._times = pd.read_csv(self.fn_times).set_index(sc.CASE)
        else:
            # First day of the event log, for day differences
            first_day = event_log.timestamp.min().date()

            # Create dataframe
            self._times = pd.DataFrame(index=pd.Index(data=event_log[sc.CASE].unique(), name=sc.CASE))

            # Note:
            # - T_TRAINING is the time at which a case can be used for training. This is the moment the target activity
            #   occurs (because then the GT is known)
            # - T_PREDICTING is the time at which a case can be used for predicting. This is the moment the prediction
            #   activity occurs (because then the
            # - The below is correct :)

            for (time_name, activity_type) in zip([sc.T_PREDICTION, sc.T_TARGET], ['prediction', 'target']):
                att = event_log[event_log[f'is_{activity_type}_activity']].set_index(sc.CASE)[sc.TIMESTAMP].dt.date
                self._times[time_name] = (att - first_day).apply(lambda x: x.days)

            # Check all prediction points are before target points
            assert (self._times[sc.T_PREDICTION] < self._times[sc.T_TARGET]).all(), \
                'The training activity for some case(s) happens after the prediction activity'

            # Save it
            self._times.to_csv(self.fn_times)

        # Prepare Activity Frequencies =================================================================================
        if self.fn_activity_frequency.exists():
            # Load the data
            self._activity_frequencies = pd.read_csv(self.fn_activity_frequency).set_index(sc.CASE)
        else:
            # Get the time at which a prediction is made, and as such when the prefix ends
            prediction_time_per_case = event_log.loc[event_log[f'is_prediction_activity'], [sc.TIMESTAMP, sc.CASE]] \
                .rename(columns={sc.TIMESTAMP: 'prefix_end_time'})
            # Add this time as event attribute
            event_log = event_log.merge(right=prediction_time_per_case, on=sc.CASE)

            # Filter on only those events that happen up to the prefix end time
            prefix_event_log = event_log[event_log[sc.TIMESTAMP] <= event_log['prefix_end_time']]

            # Compute the frequencies
            self._activity_frequencies = prefix_event_log.groupby([sc.CASE, sc.ACTIVITY]).size().unstack().fillna(
                value=0).astype(int)

            # Save the frequencies
            self._activity_frequencies.to_csv(self.fn_activity_frequency)

    # Times ============================================================================================================
    @property
    def fn_times(self) -> Path:
        """
        File containing the time values.

        Returns
        -------
        fn: Path
            Location where the time values are stored.

        """
        fn = f'times.csv'
        return self.fd_common_data / fn

    @property
    def fd_common_data(self):
        return self.fd_common / \
               self.settings.dataset / \
               f'{self.settings.prediction_activity}_{self.settings.target_activity}'

    @property
    def times(self):
        """
        DataFrame with for each case the prediction point time and the target point time

        Returns
        -------
        times: pd.DataFrame
            Time values. Index = case, columns = [T_PREDICTION, T_TARGET]

        """
        if self._times is None:
            if self.fn_times.exists():
                # Import if done before
                self._times = pd.read_csv(self.fn_times).set_index(sc.CASE)
            else:
                # Compute
                self.prepare_aft()
        return self._times

    # Activity frequencies =============================================================================================
    @property
    def fn_activity_frequency(self) -> Path:
        """
        File containing the frequencies of the activities.

        Returns
        -------
        fn: Path
            Location where the activity frequencies are stored
        """
        return self.fd_data / f'frequencies[{self.settings.prediction_activity}].csv'

    @property
    def activity_frequencies(self):
        """
        DataFrame with for each case the activity frequencies at prediction point time.

        Returns
        -------
        times: pd.DataFrame
            Activity frequencies. Index = case, columns = Activities, value[i,j] = Frequency of activity j for case i at
            the prediction point
        """
        if self._activity_frequencies is None:
            if self.fn_activity_frequency.exists():
                # import activity frequencies
                self._activity_frequencies = pd.read_csv(self.fn_activity_frequency).set_index(sc.CASE)
            else:
                # compute activity frequencies
                self.prepare_aft()
        return self._activity_frequencies

    # Case features ====================================================================================================
    @property
    def fn_encoded_cases(self) -> Path:
        """
        File containing the encoded cases

        Returns
        -------
        fn: Path
            Location where the encoded case features are stored

        """
        return self.fd_data / 'ohe_cases.csv'

    @property
    def case_features_non_ohe(self):
        if self._case_features_non_ohe is None:
            self._case_features_non_ohe = pd.read_csv(self.fn_cases).set_index(sc.CASE)
        return self._case_features_non_ohe

    @property
    def case_features_ohe(self):
        """
        Case feature values

        Returns
        -------
        case_features: pd.DataFrame
            Case feature values. Index = case, columns = encoded values. value[i,j] = feature value j for case i
        """
        if self._case_features_ohe is None:
            if not self.fn_encoded_cases.exists():
                # Compute case features

                # Select categorical/numeric features
                case_features_object = self.case_features_non_ohe.select_dtypes(include=object)
                case_features_numeric = self.case_features_non_ohe.select_dtypes(exclude=object)

                # One-hot encode the categorical features
                ohe = OneHotEncoder(sparse=False)
                data = ohe.fit_transform(case_features_object)

                # Create dataframe from One-hot encoded features
                case_features_object_ohe = pd.DataFrame(index=case_features_object.index,
                                                        columns=ohe.get_feature_names(case_features_object.columns),
                                                        data=data)

                # Add numeric features
                self._case_features_ohe = case_features_object_ohe.join(case_features_numeric)

                # Save the data
                self._case_features_ohe.to_csv(self.fn_encoded_cases)
            else:
                # Import the data
                self._case_features_ohe = pd.read_csv(self.fn_encoded_cases).set_index(sc.CASE)
        return self._case_features_ohe

    # Predictions and ground truth =====================================================================================
    @property
    def fn_predictions(self):
        """
        File containing the predictions

        Returns
        -------
        fn: Path
            Filename where predictions are stored
        """
        return self.fd / 'predictions.csv'

    # Clusters =========================================================================================================
    @property
    def fn_clusters(self):
        """
        Filename containing the clusters for each datapoint

        Returns
        -------
        fn: Path
            Location where the clusters are stored
        """
        fn = f'clusters_{self.settings.clustering_parameter_type}_{self.settings.clustering_parameter}.csv'
        return self.fd_common_data / f'clusters_{self.settings.clustering_method}' / fn

    @property
    def clusters(self):
        """
        The clusters for each case for training/predicting

        Returns
        -------
        clusters: pd.DataFrame
            The clusters for each case. Index=cases, Columns=[C_TRAINING, C_PREDICTING], Value[i,j] =
            training/predicting j for case i

        """
        if self._clusters is None:
            if self.fn_clusters.exists():
                # Import clusters
                self._clusters = pd.read_csv(self.fn_clusters).set_index(sc.CASE)
            else:
                # Create empty frame
                self._clusters = pd.DataFrame(index=self.case_features_non_ohe.index,
                                              columns=[sc.C_TRAINING, sc.C_PREDICTING])
        return self._clusters

    # Case identifier methods ==========================================================================================
    def _get_case_ids(self, t: int, is_training: bool) -> pd.Index:
        """
        Gets the case ids that are relevant for time t and training/predicting

        Parameters
        ----------
        t: int
            The time for which to get the case ids
        is_training: bool
            Whether to get the training (true) or predicting (false) ids

        Returns
        -------
        ids: pd.Index
            The relevant case ids for given t and is_training

        """
        return self.times.index[self.times[sc.T_TARGET if is_training else sc.T_PREDICTION] == t]

    def is_correct_index(self, t: int, is_training: bool, index: pd.Index) -> bool:
        """
        Verifies whether a given index is correct for given t and is_training

        Parameters
        ----------
        t: int
            The time for which to check the case ids
        is_training: bool
            Whether to check for training (True) or predicting (False) ids
        index: pd.Index
            The index to check

        Returns
        -------
        correct: bool
            True if the given index matches the expected index, False otherwise

        """
        return (index.name == sc.CASE) and (set(self._get_case_ids(t, is_training)) == set(index))

    # Data locations ===================================================================================================
    @property
    def fd_data(self) -> Path:
        """
        The folder with all raw data of the event log for the experiment

        Returns
        -------
        fd: Path
            The folder that contains the data for this event log

        """
        return Path('results/gcs_datasets') / self.settings.dataset

    @property
    def fn_events(self) -> Path:
        """
        The filename of the event table

        Returns
        -------
        fn: Path
            Location for the event table

        """
        return self.fd_data / 'events.csv'

    @property
    def fn_cases(self) -> Path:
        """
        The filename with the case attributes

        Returns
        -------
        fn: Path
            Location for the case attribute table
        """
        return self.fd_data / 'cases.csv'

    def get_cluster_data(self, t: int, is_training: bool) -> pd.DataFrame:
        df = self._x(t, is_training, ohe=False)
        # noinspection PyTypeChecker
        return df.loc[:, df.nunique() > 1]

    def _get_clusters_or_na(self, t: int, is_training: bool) -> pd.Series:
        return self.clusters.loc[
            self._get_case_ids(t, is_training), sc.C_TRAINING if is_training else sc.C_PREDICTING].rename(sc.CLUSTER)

    def _get_clusters(self, t: int, is_training: bool) -> pd.Series:
        return self._get_clusters_or_na(t, is_training).astype(int)

    def _export_clusters(self, t: int, is_training: bool, clusters: pd.Series) -> None:
        assert self.is_correct_index(t, is_training, clusters.index)
        self.clusters.loc[clusters.index, sc.C_TRAINING if is_training else sc.C_PREDICTING] = clusters
        self.clusters.to_csv(self.fn_clusters)

    def clusters_done(self, t: int, is_training: bool) -> bool:
        return not self._get_clusters_or_na(t, is_training).isna().any()

    @property
    def timestamps(self) -> Iterable[int]:
        return sorted(set(self.times.to_numpy().flatten()))

    def _get_entity_y(self, t: int, is_training: bool) -> pd.Series:
        # Generate y_data for relevant case_ids
        case_ids = self._get_case_ids(t, is_training)
        return (self.times[sc.T_TARGET] - self.times[sc.T_PREDICTION]).loc[case_ids]

    def _x(self, t, is_training: bool, ohe: bool):
        """
        Get the x data

        Parameters
        ----------
        t: int
            Time for which to get the data
        is_training: bool
            Whether the data is for training or predicting
        ohe: bool
            Whether the data needs to be one-hot-encoded or not

        Returns
        -------
        x: pd.DataFrame
            DataFrame with x data for the given parameters
        """
        # Relevant case ids
        case_ids = self._get_case_ids(t, is_training)

        # Case features Based on one-hot-encoding
        if ohe:
            x1 = self.case_features_ohe.loc[case_ids, :]
        else:
            x1 = self.case_features_non_ohe.loc[case_ids, :]

        # Activity frequencies
        x2 = self.activity_frequencies.loc[case_ids, :]

        # Return combined
        return x1.join(x2)

    def _get_cluster_xy(self, t: int, is_training: bool):
        x = self._x(t, is_training, ohe=True)
        y = self._get_entity_y(t, is_training)

        # Get clusters for these clusters
        clusters = self._get_clusters(t, is_training)

        # Return aggregated data
        return x.groupby(clusters).mean(), y.groupby(clusters).mean()

    def get_training_cluster_xy(self, t: int) -> (pd.DataFrame, pd.Series):
        return self._get_cluster_xy(t, True)

    def get_predicting_cluster_x_and_index(self, t: int):
        df = self._get_cluster_xy(t, False)[0]
        return df.to_numpy(), df.index

    def get_cluster_ground_truth(self, t: int):
        sr = self._get_cluster_xy(t, False)[1]
        sr.name = sc.CLUSTER_GROUND_TRUTH
        sr.index.name = sc.CLUSTER
        return sr

    @property
    def settings_class(self):
        return EventLogCSSettings

    @property
    def settings(self) -> EventLogCSSettings:
        return self._settings

    @property
    def fd_base(self):
        return EVENT_LOG_ROOT

    def get_entity_ground_truth(self, t: int) -> pd.Series:
        return self._get_entity_y(t, is_training=False)

    def cleanup(self):
        super().cleanup()
        file_functions.delete(self.fd_models)

    @staticmethod
    def compute_results_df(y_true: pd.Series, y_pred: pd.Series, is_cluster: bool, y_prev: [pd.Series, None] = None):

        assert y_true.isna().sum() == 0
        return super(EventLogDataHandler, EventLogDataHandler).compute_results_df(y_true, y_pred, is_cluster, y_prev)
