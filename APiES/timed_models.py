import abc
from pathlib import Path
from typing import Collection

import numpy as np

from functions.file_functions import list_files, list_dirs


class AbstractTimedModel(abc.ABC):

    @property
    @abc.abstractmethod
    def _trained_times(self) -> Collection[int]:
        pass

    # Training =========================================================================================================
    def train(self, time: int, x, y) -> None:
        self._train(time, x, y)

    @abc.abstractmethod
    def _train(self, time: int, x, y) -> None:
        pass

    # Predicting =======================================================================================================
    @abc.abstractmethod
    def predict(self, time: int, x: np.array) -> np.array:
        pass

    # Most recent model ================================================================================================
    def _most_recent_model_time(self, time: int) -> [None, int]:
        if len(self._trained_times) == 0:
            return None
        return int(max(filter(lambda x: x <= time, self._trained_times)))

    # Checks ===========================================================================================================
    def training_done(self, time: int) -> bool:
        return time in self._trained_times

    def is_trained(self, time: int) -> bool:
        return not (self._most_recent_model_time(time) is None)

    # Model properties =================================================================================================
    @property
    @abc.abstractmethod
    def needs_numerical(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def needs_sequence(self) -> bool:
        pass


class TestModel(AbstractTimedModel):
    """
    This model is only used for testing purposes. It will remember the mean value of the training data, and return
    it on prediction.
    """

    def __init__(self):
        super().__init__()
        self._means = dict()

    @property
    def _trained_times(self) -> Collection[int]:
        return self._means.keys()

    def _train(self, time, x, y):
        self._means[time] = y.mean()

    def predict(self, time: int, x: np.array) -> np.array:
        return [self._means[self._most_recent_model_time(time)]] * x.shape[0]

    @property
    def needs_numerical(self) -> bool:
        return False

    @property
    def needs_sequence(self) -> bool:
        return False


class FileBasedModel(AbstractTimedModel):

    def __init__(self, fd):
        super().__init__()
        self.fd = Path(fd)

    def fn_model(self, time: int):
        return self.fd / f'{time}.model'

    @property
    @abc.abstractmethod
    def _model_is_saved_as_folder(self):
        pass

    @property
    def _trained_times(self) -> Collection[int]:
        if not self.fd.exists():
            return []

        def m(fn: Path):
            return int(fn.name.replace('.model', ''))

        if self._model_is_saved_as_folder:
            list_of_model_paths = list_dirs(self.fd)
        else:
            list_of_model_paths = list_files(self.fd)

        return list(map(m, list_of_model_paths))

    @abc.abstractmethod
    def _cold_start_train(self, x, y):
        """
        Train a model from a cold start

        Parameters
        ----------
        x:
            feature data
        y:
            ground truth

        Returns
        -------
        model:
            trained model
        """
        pass

    @abc.abstractmethod
    def _warm_start_train(self, model, x, y):
        """
        Update an existing model

        Parameters
        ----------
        model:
            Existing model that is to be updated
        x:
            Feature data to train on
        y:
            Ground truth data to train on

        Returns
        -------

        """
        pass

    def _import_model(self, fn):
        """
        Import a model from a given file fn

        Parameters
        ----------
        fn : Path
            location of the model

        Returns
        -------
        model:
            Model saved at fn
        """
        import pickle
        assert Path(fn).exists()
        with open(fn, 'rb') as f:
            return pickle.load(f)

    def _export_model(self, model, fn):
        """
        Export a model to a given location

        Parameters
        ----------
        model:
            The model to be exported
        fn: Path
            Location where to save the given model
        """
        import pickle
        with open(fn, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    def _train(self, time: int, x, y) -> None:
        previous_time = self._most_recent_model_time(time)
        if previous_time is None:
            model = self._cold_start_train(x, y)
        else:
            model = self._import_model(self.fn_model(previous_time))
            model = self._warm_start_train(model, x, y)
        fn_out = self.fn_model(time)
        fn_out.parent.mkdir(exist_ok=True, parents=True)
        self._export_model(model, fn_out)

    def predict(self, time: int, x: np.array) -> np.array:
        model = self._import_model(self.fn_model(self._most_recent_model_time(time)))
        return self._predict_from_model(model, x)

    @abc.abstractmethod
    def _predict_from_model(self, model, x: np.array) -> np.array:
        pass


class TaxTimedModel(FileBasedModel):

    @property
    def _model_is_saved_as_folder(self):
        return True

    @staticmethod
    def _convert_xy_to_generators(x: np.array, y: np.array):
        from consumer_cluster_streaming.lstm.utils import xy_to_generators
        return xy_to_generators(x=x, y=y, batch_size=min(x.shape[0], 1000), random_state=0)

    def _cold_start_train(self, x: np.array, y: np.array):
        from consumer_cluster_streaming.lstm.tax import train_new_tax_model
        train_g, validation_g = self._convert_xy_to_generators(x, y)
        return train_new_tax_model(n_lab=1, fn_temp=None, training_sequence=train_g, validation_sequence=validation_g)

    def _warm_start_train(self, model, x: np.array, y: np.array):
        from consumer_cluster_streaming.lstm.tax import train_tax_model
        train_g, validation_g = self._convert_xy_to_generators(x, y)
        return train_tax_model(model=model, fn_temp=None, training_sequence=train_g, validation_sequence=validation_g)

    def _import_model(self, fn: Path):
        from tensorflow.python.keras.models import load_model
        return load_model(fn)

    def _export_model(self, model, fn: Path):
        model.save(fn)

    def _predict_from_model(self, model, x: np.array) -> np.array:
        return model.predict(x).flatten()

    @property
    def needs_numerical(self) -> bool:
        return True

    @property
    def needs_sequence(self) -> bool:
        return True


class AHOT(FileBasedModel):

    @property
    def _model_is_saved_as_folder(self):
        return False

    def _predict_from_model(self, model, x: np.array) -> np.array:
        return model.predict(np.array(x))

    def _cold_start_train(self, x: np.array, y: np.array):
        from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
        model = HoeffdingAdaptiveTreeRegressor()
        model.fit(np.array(x), np.array(y))
        return model

    def _warm_start_train(self, model, x: np.array, y: np.array):
        model.fit(np.array(x), np.array(y))
        return model

    @property
    def needs_numerical(self) -> bool:
        return False

    @property
    def needs_sequence(self) -> bool:
        return False
