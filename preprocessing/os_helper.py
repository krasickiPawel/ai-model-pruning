import os


def create_dir_if_not_exist(dir_path_to_create):
    if not os.path.exists(dir_path_to_create):
        os.makedirs(dir_path_to_create)

