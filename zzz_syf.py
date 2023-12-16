# from sklearn.model_selection import train_test_split
#
#
# def train_validate_split(files):
#     files, files_to_test_model = train_test_split(files, test_size=.01, random_state=42)
#     return files, files_to_test_model
#
#
# def train_test_split_depends_on_dataset(files, test_dir, train_dir, no_cancer_dir, cancer_dir, second_cancer_dir=None):
#     if test_dir is not None:
#         test_files_no_cancer = [file_path for file_path in files if test_dir in file_path and no_cancer_dir in file_path]
#         test_files_cancer = [file_path for file_path in files if test_dir in file_path and cancer_dir in file_path]
#
#         train_files_no_cancer = [file_path for file_path in files if train_dir in file_path and no_cancer_dir in file_path]
#         train_files_cancer = [file_path for file_path in files if train_dir in file_path and cancer_dir in file_path]
#
#     else:
#         no_cancer_files = [file_path for file_path in files if no_cancer_dir in file_path]
#         if second_cancer_dir is None:
#             cancer_files = [file_path for file_path in files if cancer_dir in file_path]
#         else:
#             cancer_files = [file_path for file_path in files if cancer_dir in file_path or second_cancer_dir in file_path]
#
#         train_files_no_cancer, test_files_no_cancer = train_test_split(no_cancer_files, test_size=.2, random_state=42)
#         train_files_cancer, test_files_cancer = train_test_split(cancer_files, test_size=.2, random_state=42)
#
#     print("Zdrowe: ", len(test_files_no_cancer) + len(train_files_no_cancer))
#     print("Chore: ", len(test_files_cancer) + len(train_files_cancer))
#     return train_files_no_cancer, train_files_cancer, test_files_no_cancer, test_files_cancer
#
#
# def get_datasets_from_dataset(dataset_dir_name):
#     zip_file_path = f"{datasets_dir}\\{dataset_dir_name}\\{zip_file_name}"
#     files = get_zip_files(zip_file_path)
#     # files, files_to_test_model = train_validate_split(files)
#     return train_test_split_depends_on_dataset(files, *dataset_dir_names.get(dataset_dir_name))
#
#
# def get_preprocessed_dataset_and_labels(dataset_dir_name):
#     dataset_with_labels = []
#     train_files_no_cancer, train_files_cancer, test_files_no_cancer, test_files_cancer = get_datasets_from_dataset(dataset_dir_name)
#
#     for file in train_files_no_cancer:
#         dataset_with_labels.append((file, "ok"))
#
#     for file in train_files_cancer:
#         dataset_with_labels.append((file, "cancer"))
#
#     for file in test_files_no_cancer:
#         dataset_with_labels.append((file, "ok"))
#
#     for file in train_files_cancer:
#         dataset_with_labels.append((file, "test_files_cancer"))
#
#     return dataset_with_labels
#
#
# # get_datasets_from_dataset(dataset_names[0])
# # get_preprocessed_dataset_and_labels(dataset_names[0])
#
#
# # def test_all_datasets():
# #     for dataset_dir_name in dataset_names:
# #         zip_file_path = f"{datasets_dir}\\{dataset_dir_name}\\{zip_file_name}"
# #
# #         train_files_no_cancer, train_files_cancer, test_files_no_cancer, test_files_cancer = \
# #             train_test_split_depends_on_dataset(zip_file_path, *dataset_dir_names.get(dataset_dir_name))
# #
# #         print(f"test_files_no_cancer", len(test_files_no_cancer))
# #         print(test_files_no_cancer[:10])
# #
# #         print(f"test_files_cancer", len(test_files_cancer))
# #         print(test_files_cancer[:10])
# #
# #         print(f"train_files_no_cancer", len(train_files_no_cancer))
# #         print(train_files_no_cancer[:10])
# #
# #         print(f"train_files_cancer", len(train_files_cancer))
# #         print(train_files_cancer[:10])
#
#
# # BHI   są zdrowe
# # zip -> folder "IDC_regular_ps50_idx5" -> foldery -> w każdym folder 0 i folder 1 -> pliki .png
#
# # BKH400X
# # zip -> folder "BreaKHis 400X" -> folder test i folder train -> w każdym folder benign i folder malignant -> pliki .png
#
# # BreastUltrasoundImagesDataset(BUSI)  są zdrowe
# # zip -> folder "Dataset_BUSI_with_GT" -> folder benign, folder malignant i folder normal -> pliki .png
#
# # UBIFBC
# # zip -> folder "ultrasound breast classification" -> folder train i folder val -> folder benign i folder malignant -> pliki .png
