# Print a message indicating the start of the script
print("Starting Up csv_to_sqlite.py")

# Import required libraries
import pandas as pd  # For data manipulation and analysis (handling CSV files, etc.)
import sqlite3       # For SQLite database management (creating and managing SQLite databases)
import numpy as np   # For numerical operations (often used with pandas)
import datetime      # For handling date and time (e.g., for logging with timestamps)
import argparse      # Import argparse for command-line argument parsing (to handle input arguments)
import yaml          # For parsing YAML files (commonly used for configuration files)
import json          # For parsing JSON files (useful for handling JSON data)
import os            # For interacting with the operating system (like handling file paths)

# Print a message indicating the script is defining classes
print("Defining Classes")

# Define a Logger class to handle logging operations
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

def convert_to_unix_time(date_str):
    """
    Convert a date string to Unix time.
    :param date_str: A string representing a date in 'Mon-Year' format (e.g., 'Dec-2015').
    :return: Unix time as an integer.
    """
    # Convert the date string to a pandas datetime object, then to Unix time,
    # and return the Unix time as an integer (dividing by 10**9 converts nanoseconds to seconds)
    return pd.to_datetime(date_str).value // 10**9

def cast_proper_type(column):
    """
    Determine the most appropriate data type for a DataFrame column and cast it to that type.
    :param column: A pandas Series representing a DataFrame column.
    :return: The column cast to either boolean, integer, float, or string type.
    """
    # Check if all values in the column can be converted to boolean,
    # including a special case for 'Y' and 'N' values
    if all(val in [0, 1, '0', '1', 'Y', 'N', True, False] for val in column):
        return column.replace({'Y': True, 'N': False}).astype(bool)

    # Check if all values in the column can be converted to integers
    if all(pd.to_numeric(column, errors='coerce').notnull()) and (pd.to_numeric(column).dropna() % 1 == 0).all():
        return pd.to_numeric(column, downcast='integer')

    # Check if all values in the column can be converted to floats
    if all(pd.to_numeric(column, errors='coerce').notnull()):
        return pd.to_numeric(column, downcast='float')

    # If none of the above conditions are met, convert all values to strings
    return column.astype(str)

def map_dtype_to_sqlite(col_type):
    """
    Map a pandas data type to an SQLite data type.
    :param col_type: String representation of a pandas data type.
    :return: Corresponding SQLite data type as a string.
    """
    # Check if the data type is an integer or boolean, and map to SQLite 'INTEGER'
    if col_type.startswith('int') or col_type == 'bool':
        return 'INTEGER'
    # Check if the data type is a float, and map to SQLite 'REAL'
    elif col_type.startswith('float'):
        return 'REAL'
    # Default case, particularly for 'object' and other unhandled types, mapped to SQLite 'TEXT'
    else:
        return 'TEXT'

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

print("Reading Command Line Arguments")

# Initialize an argument parser for command line argument parsing
parser = argparse.ArgumentParser(description='Process CSV file to SQLite database.')

# Add an argument for the configuration file path
# This argument is required and takes a string as input
parser.add_argument('--config', type=str, required=True, help='Path to the config file', dest="config_file_path")

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
csv_file = config['data']['input_csv']  # Path to the input CSV file
sqlite_file = config['data']['output_sqlite']  # Name of the output SQLite file
tag = config['base']['tag']  # Tag used for organizing outputs
full_column_names_file = config['data']['column_names_file']  # Path to the file with full column names
column_descriptions_file = config['data']['column_descriptions_file']  # Path to the file with column descriptions

# Construct the full path for the SQLite file
sqlite_file = os.path.join(f'outputs/{tag}/data/{sqlite_file}')

# Create necessary directories and print the result
print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/data/')))
print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/log/')))
print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/figure/')))
print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/stats/')))
print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/reports/')))
print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/models/')))


print("Initializing Logger")

# Initialize a Logger instance using the configuration
logger = Logger(config)

logger.log("----------------------Logger Initialized for csv_to_sqlite.py----------------------")

logger.log("Reading Columns Full Names")

# Open and read the full column names file using JSON
with open(full_column_names_file, 'r') as file:
    column_full_names = json.load(file)
    
# Log a message about reading column descriptions
logger.log("Reading Columns Descriptions")

# Open and read the column descriptions file using JSON
with open(column_descriptions_file, 'r') as file:
    column_descriptions = json.load(file)

# Log a message about loading the data CSV file
logger.log('Loading data CSV')

# Load the CSV file into a pandas DataFrame and replace 'nan' string with None
df = pd.read_csv(csv_file, low_memory=False).replace('nan', None)

# Log a message about loading metadata
logger.log('Loading Metadata')

# Extract metadata from the last two rows of the dataset
# Specifically, store the last two rows' 'id' values as metadata
metadata = df.iloc[-2:]['id'].values

# Remove the last two rows from the DataFrame as they contain metadata
df = df.iloc[:-2]

# Print a message to log that the script is now finding date columns
logger.log('Finding Date Columns')

# Initialize an empty list to hold the names of columns that contain date information
dates_columns = []

# Iterate through each column in the DataFrame, sorted alphabetically
for column in sorted(df.columns):
    # Drop NaN values from the column and get unique values
    samples = df[column].dropna().unique()

    # Check if the column has any data and if the first sample is in the expected date format (e.g., 'Dec-2015')
    if len(samples) > 0 and len(str(samples[0])) == 8 and str(samples[0])[3] == '-':
        # If the conditions are met, add the column name to the dates_columns list
        dates_columns += [column]

# Log a message indicating the start of UNIX timestamp conversion for date columns
logger.log('Creating UNIX Timestamp Columns for Dates')

# Iterate through each identified date column
for column in dates_columns:
    # Log the conversion of the current column
    logger.log(f"Converting column {column} to UNIX time")

    # Convert the date column to pandas datetime format, handling errors
    df[f'{column}_dt'] = pd.to_datetime(df[column], format='%b-%Y', errors='coerce')

    # Fill NaN values with a default timestamp (Unix epoch start)
    df[f'{column}_dt'].fillna(pd.Timestamp('1970-01-01'), inplace=True)

    # Calculate the UNIX timestamp by subtracting the Unix epoch start and converting to seconds
    df[f'{column}_unix'] = (df[f'{column}_dt'] - pd.Timestamp('1970-01-01')).dt.total_seconds().astype(int)

# Log a message about creating new interesting columns based on transformations
logger.log('Creating interesting columns')

# Create a new column 'term_months' by converting 'term' column values to integers
# If the value is 'nan', set it to -1
df['term_months'] = df['term'].astype('str').apply(lambda x: int(x[1:3]) if x != 'nan' else -1)

# Create a new column 'emp_length_years' by converting 'emp_length' column values to integers
# Replace '+' with an empty string and set -1 for '< 1 year' or 'nan'
df['emp_length_years'] = df['emp_length'].astype('str').apply(lambda x: int(x.split(' ')[0].replace('+','')) if x != '< 1 year' and x != 'nan' else -1)

# Assign a new unique ID to each row in the DataFrame
df['id'] = range(len(df))

# Log a message indicating the start of dropping uninteresting columns
logger.log("Dropping uninteresting columns")

# Define a list of columns to drop from the DataFrame
drop_columns = ['member_id', 'term', 'emp_length'] + \
                dates_columns + \
                [f'{col}_dt' for col in dates_columns]

# Iterate through each column in the drop_columns list
for column in drop_columns:
    # Log the dropping of the current column
    logger.log(f'Dropping column {column}')

    # Check if the column exists in the DataFrame and drop it
    if column in df.columns:
        df.drop(column, axis='columns', inplace=True)

# Log a message about starting to fill NaN values in specific columns with -1
logger.log("Filling NaNs on some columns")

# Define a list of columns where NaN values will be filled with -1
fillna_neg_one_columns = ['tot_coll_amt', 'tot_cur_bal', 'all_util', 'annual_inc_joint', 'bc_open_to_buy',
                         'deferral_term', 'collection_recovery_fee', 'hardship_last_payment_amount',
                         'hardship_payoff_balance_amount', 'max_bal_bc', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
                         'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_last_delinq', 'mths_since_last_major_derog',
                         'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
                         'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'revol_bal_joint',
                         'sec_app_fico_range_high', 'sec_app_fico_range_low', 'sec_app_mort_acc', 
                         'sec_app_mths_since_last_major_derog', 'settlement_amount', 'settlement_percentage',
                         'settlement_term', 'zip_code', 'total_rev_hi_lim', 'tot_hi_cred_lim', 'total_bc_limit',
                         'total_il_high_credit_limit']

# Iterate through each column in the fillna_neg_one_columns list
for column in fillna_neg_one_columns:
    # Log the filling of NaNs in the current column with -1
    logger.log(f'Filling NaNs in column {column} with -1')

    # Fill NaN values in the column with -1
    df[column].fillna(-1, inplace=True)

# Define a list of columns where NaN values will be filled with 0
fillna_zero_columns = ['acc_now_delinq', 'acc_open_past_24mths', 'annual_inc', 'avg_cur_bal',
                       'chargeoff_within_12_mths', 'collections_12_mths_ex_med', 'delinq_2yrs',
                       'delinq_amnt', 'hardship_amount', 'hardship_dpd', 'hardship_length',
                       'inq_fi', 'inq_last_12m', 'inq_last_6mths', 'mort_acc', 'num_accts_ever_120_pd',
                       'num_actv_bc_tl', 'num_actv_rev_tl', 'num_tl_120dpd_2m', 'open_acc_6m',
                       'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m',
                       'orig_projected_additional_accrued_interest', 'sec_app_chargeoff_within_12_mths',
                       'sec_app_collections_12_mths_ex_med', 'sec_app_inq_last_6mths', 'sec_app_num_rev_accts',
                       'sec_app_open_acc', 'sec_app_open_act_il', 'total_bal_il', 'total_cu_tl', 'open_acc',
                       'pub_rec', 'total_acc', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
                       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
                       'num_tl_op_past_12m', 'pub_rec_bankruptcies', 'tax_liens', 'total_bal_ex_mort']

# Iterate through each column in the fillna_zero_columns list
for column in fillna_zero_columns:
    # Log the filling of NaNs in the current column with 0
    logger.log(f'Filling NaNs in column {column} with 0')

    # Fill NaN values in the column with 0
    df[column].fillna(0, inplace=True)
# Define a list of column names to fill NaN (Not a Number) values with 1.
fillna_one_columns = ['bc_util', 'dti', 'dti_joint', 'il_util', 'revol_util', 'sec_app_revol_util']
for column in fillna_one_columns:
    # Log the action of filling NaNs for each column in the list.
    logger.log(f'Filling NaNs in column {column} with 1')
    # Replace NaN values in each specified column with 1.
    df[column].fillna(1, inplace=True)
    
# Define a list of column names to fill NaN values with 'N'.
fillna_N_columns = ['debt_settlement_flag', 'hardship_flag']
for column in fillna_N_columns:
    # Log the action of filling NaNs for each column in the list.
    logger.log(f'Filling NaNs in column {column} with \'N\'')
    # Replace NaN values in each specified column with 'N'.
    df[column].fillna('N', inplace=True)
    
# Define a list of column names to fill NaN values with an empty string.
fillna_empty_string_columns = ['desc', 'emp_title', 'addr_state', 'application_type', 'disbursement_method',
                              'hardship_loan_status', 'hardship_reason', 'hardship_status', 'settlement_status',
                              'title', 'verification_status_joint', 'hardship_type']
for column in fillna_empty_string_columns:
    # Log the action of filling NaNs for each column in the list.
    logger.log(f'Filling NaNs in column {column} with an empty string')
    # Replace NaN values in each specified column with an empty string.
    df[column].fillna('', inplace=True)
    
# Define a list of column names to fill NaN values with 100.
fillna_100_columns = ['pct_tl_nvr_dlq', 'percent_bc_gt_75']
for column in fillna_100_columns:
    # Log the action of filling NaNs for each column in the list.
    logger.log(f'Filling NaNs in column {column} with 100')
    # Replace NaN values in each specified column with 100.
    df[column].fillna(100, inplace=True)

# Define a list of columns to drop if they contain NaN values.
dropna_columns = ['fico_range_high', 'fico_range_low', 'funded_amnt', 'funded_amnt_inv', 'grade']
# Log the action of dropping columns with NaN values.
logger.log(f'Dropping columns: {dropna_columns}')
# Drop rows from the dataframe where any of the specified columns contain NaN values.
df.dropna(subset=dropna_columns, axis=0, inplace=True)   

# Log the modification of columns.
logger.log("Modifying Columns")

# Modify the 'emp_title' column: convert to string, convert to lower case, strip leading/trailing spaces,
# replace commas with hyphens, and replace double spaces with single spaces.
df['emp_title'] = df['emp_title'].astype('str').apply(lambda x: x.lower().strip().replace(',','-').replace('  ',' '))

# Modify the 'zip_code' column: convert to string, keep only the first three characters if not an empty string,
# and then remove leading zeros.
df['zip_code'] = df['zip_code'].astype('str').apply(lambda x: x[:3] if x != '' else x)
df['zip_code'] = df['zip_code'].str.lstrip('0')

# Log the action of casting columns to their best types.
logger.log("Casting columns to Best Types")

# Iterate through each column in the dataframe and convert them to their appropriate data types.
for column in sorted(df.columns):
    # Log the conversion process for each column.
    logger.log(f'Converting {column} elements from {str(df[column].dtype)} to {str(cast_proper_type(df[column]).dtype)}')
    # Convert each column to its appropriate type.
    df[column] = cast_proper_type(df[column])

# Log the creation of a metadata dataframe.
logger.log("Creating Metadata Dataframe")

# Create a metadata dataframe from a string, splitting on ': ' and converting values to integers.
metadata = pd.DataFrame({s.split(': ')[0]:[int(s.split(': ')[-1])] for s in metadata})

# Log the creation of a descriptions dataframe.
logger.log("Creating Descriptions Dataframe")

# Create a dictionary that describes various tables and columns in the dataset.
table_descriptions = {
    'loans_data':f'Data for all the loans available in the database, downloaded as a csv, cleaned, and put into a sqlite file on 2023-12-06. Original download from https://www.kaggle.com/code/pavlofesenko/minimizing-risks-for-loan-investments. Contains columns: {", ".join(df.columns)}.',
    'metadata':f'Metadata provided on the downloaded data, providing the total amount funded in different policy codes. Contains columns: {", ".join(metadata.columns)}.',
    'descriptions':f'A table containing a written description of each available table, and column. Contains columns: loans_data, metadata, descriptions, {", ".join(df.columns)}, {", ".join(metadata.columns)}.'
}
# Extract column descriptions for the 'metadata' table from a dictionary called column_descriptions.
metadata_columns_description = column_descriptions['metadata']

# Extract column descriptions for the 'loans_data' table from column_descriptions.
loans_data_columns_description = column_descriptions['loans_data']

# Extract column descriptions for the 'descriptions' table from column_descriptions.
descriptions_columns_description = column_descriptions['descriptions']

# Initialize a dictionary to hold various attributes of tables and columns for documentation purposes.
descriptions = {'name':[], 'full_name':[], 'type':[], 'location':[], 'description':[], 'data_type':[]}

# Iterate over table descriptions and append attributes to the descriptions dictionary.
for key, value in table_descriptions.items():
    descriptions['name'] += [key]  # Append the table name.
    descriptions['full_name'] += [key]  # Append the full table name.
    descriptions['type'] += ['table']  # Specify the type as 'table'.
    descriptions['location'] += ['root']  # Specify the location as 'root'.
    descriptions['description'] += [value]  # Append the table description.
    descriptions['data_type'] += ['TABLE']  # Specify the data type as 'TABLE'.

# Iterate over the metadata_columns_description dictionary and append column attributes.
for key, value in metadata_columns_description.items():
    descriptions['name'] += [key]  # Append the column name.
    descriptions['full_name'] += [key]  # Append the full column name.
    descriptions['type'] += ['column']  # Specify the type as 'column'.
    descriptions['location'] += ['metadata']  # Specify the location as 'metadata'.
    descriptions['description'] += [value]  # Append the column description.
    descriptions['data_type'] += ['INTEGER']  # Specify the data type as 'INTEGER'.

# Iterate over loans_data_columns_description, appending attributes for columns that exist in the dataframe df.
for key, value in loans_data_columns_description.items():
    if key in df.columns:
        descriptions['name'] += [key]  # Append the column name.
        descriptions['full_name'] += [column_full_names[key]]  # Append the full column name from a separate dict.
        descriptions['type'] += ['column']  # Specify the type as 'column'.
        descriptions['location'] += ['loans_data']  # Specify the location as 'loans_data'.
        descriptions['description'] += [value]  # Append the column description.
        descriptions['data_type'] += [map_dtype_to_sqlite(str(df[key].dtype))]  # Map the dtype to SQLite format and append.

# Iterate over the descriptions_columns_description dictionary and append column attributes.
for key, value in descriptions_columns_description.items():
    descriptions['name'] += [key]  # Append the column name.
    descriptions['full_name'] += [key]  # Append the full column name.
    descriptions['type'] += ['column']  # Specify the type as 'column'.
    descriptions['location'] += ['descriptions']  # Specify the location as 'descriptions'.
    descriptions['description'] += [value]  # Append the column description.
    descriptions['data_type'] += ['TEXT']  # Specify the data type as 'TEXT'.

# Convert the descriptions dictionary into a pandas DataFrame.
descriptions = pd.DataFrame(descriptions)

# Log a message indicating the start of a process to connect to a SQLite database.
logger.log("Connecting to Database")
# Establish a connection to a SQLite database file specified by the variable sqlite_file.
conn = sqlite3.connect(sqlite_file)

# Log the start of query creation process.
logger.log("Creating Queries")

# Create a SQL query to delete the loans_data table if it already exists in the database.
drop_loans_data_query = 'DROP TABLE IF EXISTS loans_data'

# Create a SQL query to delete the metadata table if it already exists in the database.
drop_metadata_query = 'DROP TABLE IF EXISTS metadata'

# Create a SQL query to delete the descriptions table if it already exists in the database.
drop_descriptions_query = 'DROP TABLE IF EXISTS descriptions'

# Create a SQL query to create a new table for loans data.
# This includes dynamically creating column definitions using the column names and types from the DataFrame 'df'.
create_loans_data_table_query = 'CREATE TABLE loans_data (' + ', '.join([f"\"{col}\" {col_type}" for col, col_type in zip(df.columns, [map_dtype_to_sqlite(str(df[col].dtype)) for col in df.columns])]) + ')'

# Create a SQL query to create a new table for metadata.
# The columns are defined as TEXT type, using the column names from the 'metadata' DataFrame.
create_metadata_table_query = 'CREATE TABLE metadata (' + ', '.join([f"\"{col}\" TEXT" for col in metadata.columns]) + ')'

# Create a SQL query to create a new table for descriptions.
# The columns are defined as TEXT type, using the column names from the 'descriptions' DataFrame.
create_descriptions_table_query = 'CREATE TABLE descriptions (' + ', '.join([f"\"{col}\" TEXT" for col in descriptions.columns]) + ')'

# Log the start of dropping old tables and creating new ones.
logger.log("Dropping old tables and creating new ones")

# Execute the queries to drop existing tables and create new ones.
conn.execute(drop_loans_data_query)
conn.execute(create_loans_data_table_query)
conn.execute(drop_metadata_query)
conn.execute(create_metadata_table_query)
conn.execute(drop_descriptions_query)
conn.execute(create_descriptions_table_query)

# Log the start of loading data into tables.
logger.log("Loading data into tables")

# Insert data from the 'df' DataFrame into the 'loans_data' table in the SQLite database.
# Replace the table if it already exists and do not write row indices as separate column.
df.to_sql('loans_data', conn, if_exists='replace', index=False)

# Insert data from the 'metadata' DataFrame into the 'metadata' table in the SQLite database.
# Replace the table if it already exists and do not write row indices as separate column.
metadata.to_sql('metadata', conn, if_exists='replace', index=False)

# Insert data from the 'descriptions' DataFrame into the 'descriptions' table in the SQLite database.
# Replace the table if it already exists and do not write row indices as separate column.
descriptions.to_sql('descriptions', conn, if_exists='replace', index=False)

# Close the connection to the SQLite database.
conn.close()

# Log the completion of the csv_to_sqlite.py script.
logger.log("----------------------Done with csv_to_sqlite.py----------------------")
