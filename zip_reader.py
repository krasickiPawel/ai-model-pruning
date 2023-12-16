from zipfile import ZipFile


def get_zip_files(zip_file_path):
    with ZipFile(zip_file_path, 'r') as zip_file:
        return zip_file.namelist()


def get_labels(zip_file_list, no_cancer_label, cancer_label, cancer_label_2=None):
    label_list = []
    for filepath in zip_file_list:
        if no_cancer_label in filepath:
            label_list.append(no_cancer_label)
        elif cancer_label in filepath:
            label_list.append(cancer_label)
        elif cancer_label_2 in filepath:
            label_list.append(cancer_label_2)

    return label_list
