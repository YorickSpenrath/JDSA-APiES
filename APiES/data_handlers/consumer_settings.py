from ..data_handlers.abstract_gcs_settings import AbstractGCSSettings


class ConsumerCSSettings(AbstractGCSSettings):

    @property
    def _default_dict(self):
        d = super()._default_dict
        d['tau'] = 2
        return d

    def _assign(self, d):
        super()._assign(d)
        self.tau = int(self._pop_or_default(d, 'tau'))
        assert self.tau > 0
