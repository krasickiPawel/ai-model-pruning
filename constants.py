DATASET_DIR = r"D:\UM\projekt\datasets"
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

N_SPLITS = 5
N_REPEATS = 2
df_best_features = None


