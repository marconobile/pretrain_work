import os
from os.path import isfile, join
import shutil
import errno


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



def silentremove(filename):
  try:
      os.remove(filename)
  except OSError as e: # this would be "except OSError, e:" before Python 2.6
      if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
          raise # re-raise exception if a different error occurred