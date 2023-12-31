import dataclasses
import settings


@dataclasses.dataclass
class DatasetInfo:
    name: str
    zip_file_name: str
    benign_name: str = None
    malignant_name: str = None
    normal_name: str = None


DATASETS_INFOS = [
    DatasetInfo(
        name="BreaKHis400X",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        benign_name="benign",
        malignant_name="malignant"
    ),
    DatasetInfo(
        name="BreastUltrasoundImagesDataset(BUSI)",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        benign_name="benign",
        malignant_name="malignant",
        normal_name="normal"
    ),
    DatasetInfo(
        name="UltrasoundBreastImagesforBreastCancer",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        benign_name="benign",
        malignant_name="malignant"
    ),
    DatasetInfo(
        name="BreastHistopathologyImages",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        malignant_name="class1",
        normal_name="class0"
    )
]
DATASETS_INFOS.pop(-1)
