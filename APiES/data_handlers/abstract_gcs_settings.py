import abc

from functions.alm import AbstractSettings
from APiES.constants import CLUSTERING_PARAMETER_TYPE, MODEL_TYPE, CLUSTERING_METHOD, \
    CLUSTERING_PARAMETER


class AbstractGCSSettings(AbstractSettings, abc.ABC):
    CLUSTER_COUNT = 'cluster_count'  # Define number of clusters
    CLUSTER_SIZE = 'cluster_size'  # Define size of clusters
    ALL_CLUSTERING_PARAMETER_TYPES = [CLUSTER_SIZE, CLUSTER_COUNT]

    NATIVE = 'native'  # as defined in the paper
    RANDOM = 'random'  # random cluster assignment
    ALL_CLUSTERING_METHODS = [NATIVE, RANDOM]

    def _assign(self, d):
        # Definition of the number of clusters
        self.clustering_parameter_type = self._pop_or_default(d, CLUSTERING_PARAMETER_TYPE)
        assert self.clustering_parameter_type in AbstractGCSSettings.ALL_CLUSTERING_PARAMETER_TYPES
        self.clustering_parameter = self.int_or_none(self._pop_or_default(d, CLUSTERING_PARAMETER))

        # Definition of the clustering features
        self.clustering_method = self._pop_or_default(d, CLUSTERING_METHOD)
        assert self.clustering_method in AbstractGCSSettings.ALL_CLUSTERING_METHODS

        # Model used in the prediction
        self.model_type = self._pop_or_default(d, MODEL_TYPE)

    @property
    def param(self):
        if self.clustering_parameter_type == self.CLUSTER_SIZE:
            return 'rho'
        elif self.clustering_parameter_type == self.CLUSTER_COUNT:
            return 'k'
        else:
            raise NotImplementedError

    @property
    def tex_param(self):
        if self.clustering_parameter_type == self.CLUSTER_SIZE:
            return r'$\rho$'
        elif self.clustering_parameter_type == self.CLUSTER_COUNT:
            return '$k$'
        else:
            raise NotImplementedError

    @property
    def _default_dict(self):
        return dict(model_type='TestModel',
                    clustering_parameter_type=AbstractGCSSettings.CLUSTER_SIZE,
                    clustering_method=AbstractGCSSettings.NATIVE)
