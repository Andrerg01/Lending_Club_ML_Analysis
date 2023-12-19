print("Starting Up")
import pandas as pd # Data manipulation
import sqlite3 # Database connection
import numpy as np # Numerical computation
import datetime # Date manipulation
import seaborn as sns # Plotting
import matplotlib.pyplot as plt # Plotting
import os # File manipulation
import yaml # Config file parsing
import argparse # Command line argument parsing
import sys
import warnings

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SupportVectorClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real

print("Defining Classes")
# Defining a class for logging messages to a file
class Logger:
    """
    A class for logging messages to a file.

    Attributes:
        config (dict): The configuration settings.
        log_dir (str): The directory for storing log files.
        tag (str): The tag for identifying the log files.
        file_path (str): The path to the log file.
        verbose (bool): Flag indicating whether to print log messages to the console.
    """
    def __init__(self, config):
        # Constructor to initialize the Logger instance
        self.config = config  # Store the provided configuration
        self.log_dir = config['logging']['out_dir']  # Directory to output logs
        self.tag = config['base']['tag']  # Tag for the log (e.g., identifying the run)
        # Construct the file path for the log file
        self.file_path = os.path.join('outputs', self.tag, self.log_dir, 'log.txt')
        self.verbose = config['logging']['verbose']  # Verbose flag to control output
        
    def log(self, message):
        """
        Logs a message to the log file.

        Args:
            message (str): The message to be logged.
        """
        # Method to log a message
        current_datetime = datetime.datetime.now()  # Get the current date and time
        # Format the datetime as a string
        datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Format the log message with a timestamp
        log_message = f"{datetime_string}: {message}"
        if self.verbose:
            # If verbose mode is on, print the log message to the console
            print(log_message)
        with open(self.file_path, "a") as f:
            # Open the log file in append mode and write the log message
            f.write(f'{log_message}\n')
            
print("Defining Functions")
def bestbandwidth(data):
    """
    Calculate the optimal bandwidth for kernel density estimation using the Silverman's rule of thumb.

    Parameters:
    data (array-like): The input data for which the bandwidth needs to be calculated.

    Returns:
    float: The optimal bandwidth value.

    """
    return 1.06*np.std(data)*len(data)**(-1/5)

def transform_type(sqlite_type):
    """
    Transforms the SQLite data type to a Python data type.

    Parameters:
    sqlite_type (str): The SQLite data type.

    Returns:
    str: The corresponding Python data type.
    """
    if sqlite_type == 'INTEGER':
        return 'int'
    if sqlite_type == 'REAL':
        return 'float'
    else:
        return 'object'

def map_dtype_to_sqlite(col_type):
    """
    Maps a column data type to the corresponding SQLite data type.

    Parameters:
        col_type (str): The column data type.

    Returns:
        str: The corresponding SQLite data type.

    """
    if col_type.startswith('int') or col_type == 'bool':
        return 'INTEGER'
    elif col_type.startswith('float'):
        return 'REAL'
    else:  # Default case, particularly for 'object' and other unhandled types
        return 'TEXT'

print("Reading Command Line Arguments")

# Initialize an argument parser for command line argument parsing
parser = argparse.ArgumentParser(description="Read configuration file.")

# Add an argument for the configuration file path
# This argument is required and takes a string as input
parser.add_argument("--config", required=True, help="Path to the config file", dest="config_file_path")

# Parse the command line arguments
args = parser.parse_args()

# Store the path to the configuration file from the parsed arguments
config_file_path = args.config_file_path


print(f"Reading Config File {config_file_path}")

# Open and read the configuration file using YAML
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

print("Defining Variables and Creating Directories")

sqlite_file = config['data']['output_sqlite']

tag = config['base']['tag']

git_repo = config['base']['git_repo']

fontsize = config['plotting']['fontsize']
figsize_x = config['plotting']['figure_xsize']
figsize_y = config['plotting']['figure_ysize']
bayes_search_iterations = config['machine_learning']['bayes_search_iterations']
random_state = config['machine_learning']['random_state']

optimization_sample_size = config['machine_learning']['optimization_sample_size']

out_dir_figures = f"outputs/{tag}/figures"
out_dir_log = f"outputs/{tag}/log"
out_dir_stats = f"outputs/{tag}/stats"


sqlite_file = f"outputs/{tag}/data/{sqlite_file}"

columns_of_interest = config['base']['columns_of_interest']

print("Initializing Logger")

# Initialize a Logger instance using the configuration
logger = Logger(config)

logger.log("----------------------Logger Initialized for machine_learing_trainings.py----------------------")

logger.log("Loading Data")
# Defining the connection to the database
conn = sqlite3.connect(sqlite_file)

# Loading data into dataframe
data_fetch_query = f"""SELECT * 
                       FROM loans_data_ML
                       ORDER BY RANDOM()"""

loans_data = pd.read_sql_query(data_fetch_query, conn, index_col='id')

# Closing connection
conn.close()

logger.log("Separating sample data for ML hyperparameter optimization.")

sample_data = loans_data.sample(optimization_sample_size)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    # Since we're setting this at the sys level, it should not be overridden
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

X, y = loans_data.drop('loan_status', axis='columns').values, loans_data['loan_status'].values

ML_columns = loans_data.drop('loan_status', axis='columns').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

