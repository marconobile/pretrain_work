import time
import datetime

def print_obj_API(obj):
  attributes = [attr for attr in dir(obj) if not attr.startswith('__')]

  # Differentiating between methods and attributes
  methods = [attr for attr in attributes if callable(getattr(obj, attr)) and not attr.startswith('_')]
  attributes = [attr for attr in attributes if not callable(getattr(obj, attr)) and not attr.startswith('_')]

  print(f"{type(obj).__name__} Attributes:", attributes)
  print("\n")
  print(f"{type(obj).__name__} Methods:", methods)



class TimeThis:
  def __init__(self, name=None):
    """
    Initialize the timer with an optional name.
    """
    self.name = name

  def __enter__(self):
    self.start_time = time.perf_counter()
    if self.name: print(f"Starting counter: {self.name} ...")

  def __exit__(self, exc_type, exc_val, exc_tb):
    end_time = time.perf_counter()
    elapsed_time = end_time - self.start_time
    # Convert elapsed time to hh:mm:ss format
    td = datetime.timedelta(seconds=elapsed_time)
    str_time = str(td)
    if self.name: print(f"Execution time of {self.name}: {str_time}")
    else: print(f"Execution time: {str_time}")