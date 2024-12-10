import os
from os.path import isfile, join
import shutil


def ls(dir):
  return [join(dir, f) for f in os.listdir(dir) if isfile(join(dir, f))]


def move_files_to_folder(dst_folder, files_to_move):
    out_filepaths = []
    for src_filepath in files_to_move:
        filename = os.path.basename(src_filepath)
        dst_filepath = join(dst_folder, filename)
        out_filepaths.append(dst_filepath)
        shutil.copy(src_filepath, dst_filepath)
    return out_filepaths


def rm_and_recreate_dir(dir):
  '''delete folder and its content'''
  if os.path.exists(dir): shutil.rmtree(dir)
  os.makedirs(dir)


def parse_smiles_file(file_path):
  """
  Parses a .smiles file and returns a list of smiles str. (or a .txt where each line is a smile)
  :param file_path: Path to the .smiles file
  :return: List of smiles strings
  """
  smiles_list = []
  with open(file_path, 'r') as file:
    for line in file:
      smiles = line.strip()
      if smiles: smiles_list.append(smiles)
  return smiles_list
