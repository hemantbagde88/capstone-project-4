import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import logging
import yaml

#Ensure that "logs" directory exists
log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setlevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(param_path: str) -> dict:
    """load the parameter from the yaml file."""
    try:
        with open(param_path, 'r') as file:
            param = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", param_path)
        return param
    except FileNotFoundError:
        logger.error("file not found: %s", param_path)
        raise 
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise 
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """load the data from the csv file"""
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to purse the CSV file: %s", e)
        raise 
    except Exception as e:
        logger.error("unexpected error occured while loading the data: %s", e)
        raise 






