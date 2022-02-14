from pathlib import Path

CASE = 'case'
BRIGHT_ORANGE = '#FF8900'
CLUSTER = 'cluster'
CLUSTER_PREDICTION = 'cluster_prediction'
CLUSTERING_PARAMETER_TYPE = 'clustering_parameter_type'
MODEL_TYPE = 'model_type'
CLUSTERING_METHOD = 'clustering_method'
CLUSTERING_PARAMETER = 'clustering_parameter'
TOTAL_VALUE = 'total_value'
consumer = 'consumer_id'
ACTIVITY = 'activity'
TIMESTAMP = 'timestamp'
CLUSTER_GROUND_TRUTH = 'cluster_ground_truth'
Y_TRUE = 'y_true'
Y_PRED = 'y_pred'
T_TARGET = 't_target'
T_PREDICTION = 't_prediction'
C_TRAINING = 'c_training'
C_PREDICTING = 'c_predicting'
IS_PREDICTING_ACTIVITY = 'is_predicting_activity'
IS_TARGET_ACTIVITY = 'is_target_activity'
EXPERIMENT = 'experiment'
METRIC = 'metric'
VALUE = 'value'
ENTITY_PREDICTION = 'entity_prediction'
NUMBER_OF_WEEKS = 'number_of_weeks'

keep_feats = ['freshness', 'mean_item_value', 'product_density', 'count', 'number_of_weeks', 'total_item_count',
              'total_value', '1', '2', '3', '4', '5', '6', '7', 'unknown']

try:
    import themes
except ModuleNotFoundError:
    class T:
        WHITE = '#FFFFFF'
        SKY_BLUE = '#00E1F0'
        FRESH_GREEN = '#60E99F'
        BRIGHT_ORANGE = '#FF8900'
        NEW_SET = [SKY_BLUE, FRESH_GREEN, BRIGHT_ORANGE]

        PLN_DARK = [146 / 255, 39 / 255, 144 / 255]
        DIV_DARK = [24 / 255, 167 / 255, 157 / 255]
        BLK_DARK = [15 / 255, 118 / 255, 187 / 255]
        OTG_DARK = [241 / 255, 89 / 255, 42 / 255]
        FLX_DARK = [217 / 255, 28 / 255, 92 / 255]
        DARK_COLOURS = [PLN_DARK, DIV_DARK, BLK_DARK, OTG_DARK, FLX_DARK]


    themes = T()

try:
    from _cdh_local import EVENT_LOG_ROOT
except (ImportError, ModuleNotFoundError):
    EVENT_LOG_ROOT = Path('results') / 'eventlogs'

try:
    from _cdh_local import CCS_ROOT
except(ImportError, ModuleNotFoundError):
    CCS_ROOT = Path('results') / 'shoppers'
