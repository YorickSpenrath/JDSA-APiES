import logging

from ..data_handlers import AbstractDataHandler


def train(adh: AbstractDataHandler, t: int):
    log = logging.getLogger(str(adh))

    # Skip if done
    if adh.model.training_done(time=t):
        log.info('Model already trained')
        return

    # Get data
    log.info('Reading Data')
    x, y = adh.get_training_cluster_xy(t=t)

    # Skip if no data
    if len(x) == 0:
        log.info('No datapoints to update model')
        return

    # Get, update and export model
    log.info('Training model')
    adh.model.train(time=t, x=x, y=y)
    log.info('Finished training')
