DATASETS_DIR = "datasets"
MODELS_DIR = "models"
PRUNING_DIR = "pruning"

NUM_CLASSES = 2

N_SPLITS = 5
# N_SPLITS = 2
N_REPEATS = 2

BATCH_SIZE = 8
# NUM_EPOCHS = 2
NUM_EPOCHS = 10

LEARNING_RATE = 0.001
MOMENTUM = 0.9

MODEL_PATH = "breast_model_pruning.pth"
PRUNING_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8]
