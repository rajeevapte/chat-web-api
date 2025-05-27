import configparser
ip_csv_files = "def_csv_input"
ip_pdf_files = "def_pdf_input"
op_vector_store = "def_store"



def read_properties_file(file_path="config.properties"):
    config = configparser.ConfigParser(allow_no_value=True)
    with open(file_path, 'r') as config_file:
        # Read the file content and add a default section if it's missing
        content = config_file.read()
        if not any(line.startswith('[') and line.endswith(']') for line in content.splitlines()):
            content = '[DEFAULT]\n' + content
        config.read_string(content)
    return config

# Usage
file_path = 'config.properties'
config = read_properties_file(file_path)

# Accessing properties
try:
    ip_csv_files = config['DEFAULT']['ip_csv_files']
    ip_pdf_files = config['DEFAULT']['ip_pdf_files']
    op_vector_store = config['DEFAULT']['op_vector_store']
    print(f"DB Host: {ip_csv_files}, DB User: {ip_pdf_files}, DB Password: {op_vector_store}")
except KeyError as e:
    print(f"Error: Property '{e}' not found in the configuration file.")