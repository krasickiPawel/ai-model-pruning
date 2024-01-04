import dataclasses
import settings


@dataclasses.dataclass
class DatasetInfo:
    name: str
    zip_file_name: str
    benign_name: str = None
    malignant_name: str = None
    normal_name: str = None
