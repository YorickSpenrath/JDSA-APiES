import logging
import sys
from datetime import datetime

from ..data_handlers import AbstractDataHandler
from ..execution_modules import training_cluster, predicting_cluster, predict, train, \
    compute_results

ps = list

logging.basicConfig(format='\t%(message)s', stream=sys.stdout, level=logging.INFO)


def run_experiment(adh: AbstractDataHandler):
    file_handler = logging.FileHandler(adh.log_file, 'a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(';%(relativeCreated)s;%(module)s;%(message)s'))
    log = logging.getLogger(str(adh))
    log.addHandler(file_handler)

    log.info(f'Started {adh.name} {datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")}')

    def set_time_step_to_logger(ti):
        if ti is None:
            ti = ''
        file_handler.setFormatter(logging.Formatter(f'{ti};%(relativeCreated)s;%(module)s;%(message)s'))

    if adh.is_done():
        log.info('Already done')
        log.info('Cleanup')
        adh.cleanup()
        log.info('Finished')
        return

    for t in ps(adh.timestamps):
        set_time_step_to_logger(t)

        if adh.clusters_done(t, True) \
                and adh.clusters_done(t, False) \
                and adh.cluster_predictions_done(t) \
                and adh.model.training_done(t):
            continue

        print(f'----------{t}---------')
        log.info(f'Started Time Step')

        # Cluster training data
        log.info(f'Clustering - Training')
        training_cluster(adh, t)

        # Creating Training x,y
        log.info(f'Training')
        train(adh, t)

        # Cluster testing data
        log.info(f'Clustering - Predicting')
        predicting_cluster(adh, t)

        # Predict the
        log.info(f'Predicting')
        predict(adh, t)

    set_time_step_to_logger('Wrap')

    log.info('Computing Results')
    compute_results(adh)
    log.info('Cleanup')
    adh.cleanup()
    log.info('Finished')
