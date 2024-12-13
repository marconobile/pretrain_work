def print_obj_API(obj):
  attributes = [attr for attr in dir(obj) if not attr.startswith('__')]

  # Differentiating between methods and attributes
  methods = [attr for attr in attributes if callable(getattr(obj, attr)) and not attr.startswith('_')]
  attributes = [attr for attr in attributes if not callable(getattr(obj, attr)) and not attr.startswith('_')]

  print(f"{type(obj).__name__} Attributes:", attributes)
  print("\n")
  print(f"{type(obj).__name__} Methods:", methods)