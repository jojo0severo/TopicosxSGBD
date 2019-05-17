import os
import pathlib


def read_csv(path):
    with open(path + '/file.txt', 'r') as csv_file:
        return csv_file.read()


def get_folders(path):
    dirs = []
    for directory in os.listdir(str(path.absolute())):
        if directory.startswith('player'):
            dirs.append(str((path / directory).absolute()))

    return dirs


def concatenate_files(folders):
    with open('dataset.csv', 'a') as dataset:
        for f in folders:
            csv_file = read_csv(f)
            dataset.write(csv_file)


if __name__ == '__main__':
    root = pathlib.Path(__file__).parent
    concatenate_files(get_folders(root))
