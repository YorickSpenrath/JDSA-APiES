import logging

import pandas as pd

from ..data_handlers import AbstractDataHandler
from .. import constants as sc


def predict(adh: AbstractDataHandler, t: int):
    log = logging.getLogger(str(adh))
    # Output file
    if adh.cluster_predictions_done(t=t):
        log.info('Predictions already done')
        return

    # Get data
    log.info('Loading data')
    x, index = adh.get_predicting_cluster_x_and_index(t=t)

    # Get model
    if not adh.model.is_trained(time=t):
        log.info('No model exists, skipping predictions')
        res = pd.Series(data=None)
    else:
        log.info(f'Using most recent model, making predictions')

        # Make predictions
        y_pred = adh.model.predict(time=t, x=x)
        res = pd.Series(data=y_pred, index=index)

    res.index.name = sc.CLUSTER
    res.name = sc.CLUSTER_PREDICTION
    log.info('Exporting predictions')
    adh.export_cluster_predictions(t=t, predictions=res)
