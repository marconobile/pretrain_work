import os

def generate_file(path, filename):
  '''
  if path does not exist it is created
  if filename must have extension, default: '.txt'
  if file already exists it is overwritten

  args:
      - path directory where to save smiles list as txt
      - filename name of the file. By default it is created a txt file
  '''
  os.makedirs(path, exist_ok=True)
  path_to_file = os.path.join(path, filename)
  filename_ext = os.path.splitext(path_to_file)[-1].lower()
  if not filename_ext: path_to_file += '.txt'
  if os.path.isfile(path_to_file):
      try: os.remove(path_to_file)
      except OSError: raise f"{path_to_file} already existing and could not be removed"
  return path_to_file


def create_log(path, name="log.txt"):
  if not name.endswith(".txt"): name += ".txt"
  generate_file(path, name)
  return os.path.join(path, name)


def append_line_to_log(path_to_log, line):
  with open(path_to_log, "a") as log: log.write(line + "\n")