DATASETS_DIR = "datasets"
MODELS_DIR = "models"
PRUNING_DIR = "pruning"

NUM_CLASSES = 2

N_SPLITS = 5
# N_SPLITS = 2
N_REPEATS = 2

BATCH_SIZE = 8
# NUM_EPOCHS = 1
NUM_EPOCHS = 10

LEARNING_RATE = 0.001
MOMENTUM = 0.9

MODEL_PATH = "breast_model_pruning.pth"
PRUNING_VALUES = [0, 0.1, 0.2, 0.4, 0.6, 0.8]

TRAINING_SCORES_NAME = "scores.npy"
PRUNING_SCORES_NAME = "pruning_scores.npy"

P_VALUE = 0.05


fresh_trained_models_dir = ""
fresh_pruned_models_dir = ""
TRAINED_MODELS_TO_PRUNE_CURRENT_DATETIME_DIR = "2024-01-03_19-30-05"
PRUNED_MODELS_TO_TEST_CURRENT_DATETIME_DIR = "2024-01-08_03-12-42"
old_trained_models_dir = "2024-01-03_01-05-25"
old_pruned_models_dir = "2024-01-03_16-40-51"
