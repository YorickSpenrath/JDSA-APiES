from ..data_handlers.abstract_gcs_settings import AbstractGCSSettings


class EventLogCSSettings(AbstractGCSSettings):

    def _assign(self, d):
        # Dataset name
        self.dataset = d.pop('dataset')

        # super() is called after the dataset, as the _default_dict requires the dataset name to be known
        super()._assign(d)

        # Activity at which a prediction is made
        self.prediction_activity = self._pop_or_default(d, 'prediction_activity')

        # Activity whose duration since the prediction activity is predicted
        self.target_activity = self._pop_or_default(d, 'target_activity')

    @property
    def _default_dict(self):
        # The default dict depends on the dataset
        if self.dataset == 'bpic2019':
            z = dict(target_activity='Record Invoice Receipt', prediction_activity='Vendor creates invoice')
        else:
            # Fall-through: another dataset is used, so the target_activity/prediction_activity are not defined by
            # default
            z = dict()

        return {**z, **super()._default_dict}
