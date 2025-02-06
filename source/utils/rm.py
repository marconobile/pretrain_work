import os
import shutil
import multiprocessing
from source.utils.parser import MyArgPrsr


parser_entries = [
    {'identifiers': ["-p", '--path'], 'type': str, 'help': 'Path from where to rm'},
]


def remove_file_or_folder(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(f"Error removing {path}: {e}")

def remove_files_and_folders_in_root(root_path):

    with os.scandir(root_path) as entries:
        # Filter non-hidden files using entry.is_file() and entry.name
        paths = [os.path.join(root_path, entry.name) for entry in entries]

    with multiprocessing.Pool(64) as pool:
        pool.map(remove_file_or_folder, paths, chunksize=50)

# example:
if __name__ == "__main__":
    args = MyArgPrsr(parser_entries)
    remove_files_and_folders_in_root(args.path)