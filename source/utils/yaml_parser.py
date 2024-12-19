import yaml

# Function to read a YAML file and convert it to a dictionary
def yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
