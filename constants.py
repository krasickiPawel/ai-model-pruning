DATASET_DIR = "datasets"
ZIP_FILE_NAME = "archive.zip"

DATASET_NAMES = [
    "BreaKHis400X",
    # "BreastUltrasoundImagesDataset(BUSI)",
    # "UltrasoundBreastImagesforBreastCancer",
    # "BreastHistopathologyImages" # dodać wybieranie tylko tego folderu wewnętrzengo żeby /2
]

DATASET_CANCER_LABELS = {
    "BreaKHis400X": ("benign", "malignant"),
    "BreastHistopathologyImages": ("class0", "class1"),
    "BreastUltrasoundImagesDataset(BUSI)": ("normal", "malignant", "benign"),
    "UltrasoundBreastImagesforBreastCancer": ("benign", "malignant")
}

NUM_CLASSES = 2
N_SPLITS = 5
N_REPEATS = 2

BATCH_SIZE = 8
NUM_EPOCHS = 3

LEARNING_RATE = 0.01
MOMENTUM = 0.9

MODEL_PATH = "breast_model_pruning.pth"
PRUNING_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8]

