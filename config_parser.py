import configparser
import json
import os
import sys

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def create_config():
    config = configparser.ConfigParser()
    
    # Add sections and key-value pairs
    config['General'] = {
        'team': 'NOVO',
        'players': [],
        'teamSorting': 'false',
        'maps': ["Haven", "Fracture", "Bind", "Ascent", "Icebox", "Split", "Breeze", "Lotus", "Pearl", "Sunset", "Abyss"]
    }

    # Get the absolute path for config.ini
    config_path = os.path.join(get_base_path(), 'config.ini')
    
    # Write the configuration to a file
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    return read_config()

def read_config():
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    
    # Get the absolute path for config.ini
    config_path = os.path.join(get_base_path(), 'config.ini')
    
    # Read the configuration file
    config.read(config_path)
    
    # Access values from the configuration file
    teamSorting = config.getboolean('General', 'teamSorting')
    team = config.get('General', 'team')
    players = json.loads(config.get('General', 'players'))
    
    # Return a dictionary with the retrieved values
    config_values = {
        'teamSorting': teamSorting,
        'team': team,
        'players': players,
    }
    
    return config_values