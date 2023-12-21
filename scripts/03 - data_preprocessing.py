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

def loan_status_to_int(status):
    """
    Converts loan status to an integer value.

    Parameters:
    status (str): The loan status.

    Returns:
    int: The corresponding integer value for the loan status.
         - 0 for 'Charged Off'
         - 1 for 'Fully Paid'
         - -1 for any other status
    """
    if status == 'Charged Off':
        return 0
    if status == 'Fully Paid':
        return 1
    else:
        return -1

def bestbandwidth(data):
    """
    Calculate the optimal bandwidth for kernel density estimation using the Silverman's rule of thumb.

    Parameters:
    data (array-like): The input data for which the bandwidth needs to be calculated.

    Returns:
    float: The optimal bandwidth value.

    """
    return 1.06*np.std(data)*len(data)**(-1/5)

def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not already exist.
    :param directory: Path of the directory to be created.
    :return: A message stating whether the directory was created or already exists.
    """
    # Check if the directory does not exist
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        return f"Created directory: {directory}"
    else:
        # Return a message stating the directory already exists
        return f"Directory already exists: {directory}"

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

# Extract various configuration settings from the config file
sqlite_file = config['data']['output_sqlite']
columns_of_interest = config['base']['columns_of_interest']
tag = config['base']['tag']

sqlite_file = f"outputs/{tag}/data/{sqlite_file}"

print("Initializing Logger")

# Initialize a Logger instance using the configuration
logger = Logger(config)

logger.log("----------------------Logger Initialized for data_preprocessing.py----------------------")

logger.log("Loading Data")

# Defining the connection to the database
conn = sqlite3.connect(sqlite_file)

# Defining the query to fetch the descriptions
description_fetch_query = f"""SELECT * FROM descriptions"""

# Loading descriptions into dataframe
descriptions = pd.read_sql_query(description_fetch_query, conn, index_col = 'name')

# Loading the type of each column for the loans_data table
column_types = {idx:transform_type(row['data_type']) for idx, row in descriptions.iterrows() if row['location'] == 'loans_data' and idx in columns_of_interest}

# Defining the query to fetch the data
data_fetch_query = f"""SELECT {', '.join(columns_of_interest)} 
                       FROM loans_data"""

# Loading the data into a dataframe
loans_data = pd.read_sql_query(data_fetch_query, conn, index_col='id', dtype=column_types)

# Closing connection
conn.close()

# Filtering out columns that are known to be bad
logger.log("Filtering known bad columns")
loans_data = loans_data[loans_data['issue_d_unix'] != 0]

# Creating new useful columns
logger.log("Creating columns")

# Creating column for issue date as datetime
loans_data['issue_d'] = pd.to_datetime(loans_data['issue_d_unix'], unit='s')
descriptions = pd.concat([descriptions, pd.DataFrame({
    'name':['issue_d'],
    'full_name': ['Issue Date'],
    'type': ['Column'],
    'location': ['loans_data'],
    'description': ['Date the loan was issued'],
    'data_type': ['TEXT']
}).set_index('name')])

# Creating column holding the month of the year the loan was issued
loans_data['issue_month'] = loans_data['issue_d'].apply(lambda x: x.month)
descriptions = pd.concat([descriptions, pd.DataFrame({
    'name':['issue_month'],
    'full_name': ['Issue Month'],
    'type': ['Column'],
    'location': ['loans_data'],
    'description': ['Month of the year the loan was issued'],
    'data_type': ['INT']
}).set_index('name')])

logger.log("Limiting dataset to two types of Loan Status only")
# Limiting dataset to two types of Loan Status only
loans_data = loans_data[(loans_data['loan_status'] == 'Charged Off') | (loans_data['loan_status'] == 'Fully Paid')]

loans_data['loan_status'] = loans_data['loan_status'].apply(lambda x: x == 'Charged Off')

logger.log("Declaring Numerical Columns")
# Columns that are numerical and will be used for ML as numbers, rather than dummies
numerical_columns = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 
                     'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                     'mort_acc', 'pub_rec_bankruptcies', ]

logger.log("Declaring Dummy Columns")
# Columns that will be converted to dummy variables (True or False)
dummy_columns = ['term_months', 'sub_grade', 'home_ownership', 'verification_status',
                 'purpose', 'initial_list_status', 'application_type']

logger.log("Declaring Columns to Drop")
# Columns that will be dropped
drop_columns = ['grade', 'issue_d', 'issue_d_unix']

logger.log("Dropping Non-Interesting Columns")
loans_data_ML = loans_data.drop(drop_columns, axis='columns')
logger.log("Creating Dummy Columns")
loans_data_ML = pd.get_dummies(loans_data_ML, columns=dummy_columns, drop_first=True)

logger.log(f"Connecting to Database at {sqlite_file}")
# Defining the connection to the database
conn = sqlite3.connect(sqlite_file)

logger.log("Creating Queries")
# Query to drop a table for the loans data suited for machine learning if it exists
drop_loans_data_ML_query = 'DROP TABLE IF EXISTS loans_data_ML'

# Query to create a table for the loans data suited for machine learning
loans_data_ML.reset_index(inplace=True)
create_loans_data_ML_table_query = 'CREATE TABLE loans_data_ML (' + ', '.join([f"\"{col}\" {col_type}" for col, col_type in zip(loans_data_ML.columns, [map_dtype_to_sqlite(str(loans_data_ML[col].dtype)) for col in loans_data_ML.columns])]) + ')'

logger.log("Dropping old tables and creating new ones")
# Drops and creates the table for the loans data suited for machine learning
conn.execute(drop_loans_data_ML_query)
conn.execute(create_loans_data_ML_table_query)

logger.log("Loading data into tables")
# Insert data from DataFrame to the SQLite table
loans_data_ML.to_sql('loans_data_ML', conn, if_exists='replace', index=False)

# Close the connection to the database
conn.close()
logger.log("Done!")