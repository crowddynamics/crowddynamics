import os

project_root = '/home/jaan/Dropbox/Projects/Crowd-Dynamics'


def default_path(fname, *folder_names):
    """

    :param root:
    :param folder_names:
    :return:
    """

    folder = os.path.join(project_root, *folder_names)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, fname)
    return filepath
