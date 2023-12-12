print("Starting Up csv_to_sqlite.py")
import pandas as pd  # For data manipulation and analysis
import sqlite3       # For SQLite database management
import numpy as np   # For numerical operations
import datetime      # For handling date and time
import time          # For time-related tasks
import argparse # Import argparse for command-line argument parsing
import datetime
import yaml
import json
import os

print("Defining Classes")
class Logger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config['logging']['out-dir']
        self.tag = config['base']['tag']
        self.file_path = os.path.join('outputs', self.tag, self.log_dir, 'log.txt')
        self.verbose = config['logging']['verbose']
        
    def log(self, message):
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{datetime_string}: {message}"
        if self.verbose:
            print(log_message)
        with open(self.file_path, "a") as f:
            f.write(f'{log_message}\n')

print("Defining Functions")

def convert_to_unix_time(date_str):
    """
    Convert a date string to Unix time.
    :param date_str: A string representing a date in 'Mon-Year' format (e.g., 'Dec-2015').
    :return: Unix time as an integer.
    """
    return pd.to_datetime(date_str).value // 10**9

def analyze_column(column):
    """
    Analyze the data type of a column in a DataFrame.
    :param column: A pandas Series representing a DataFrame column.
    :return: A string indicating the data type of the column.
    """
    # Check if all elements in the column are strings
    if all(isinstance(x, str) for x in column):
        # Check if all strings are either numeric or 'nan'
        if all(x.replace('.', '', 1).isdigit() or x.lower() == 'nan' for x in column):
            return 'REVIEW_TEXT'
        else:
            return 'TEXT'
    
    # If not all elements are strings, check other data types
    else:
        # Check if all elements are floats
        if all(isinstance(x, float) for x in column):
            # Check if floats can be converted to integers without loss of information
            if not all(x.is_integer() for x in column):
                return 'FLOAT'
            else:
                return 'REVIEW_FLOAT'
        # Check if all elements are integers
        elif all(isinstance(x, (int, np.integer)) for x in column):
            return 'INT'
        # If none of the above, return 'REVIEW' for further examination
        else:
            return 'REVIEW'
        
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return f"Created directory: {directory}"
    else:
        return f"Directory already exists: {directory}"

print("Reading Command Line Arguments")

parser = argparse.ArgumentParser(description='Process CSV file to SQLite database.')

parser.add_argument('--config', type=str, required=True, help='Path to the config file', dest="config_file_path")

args = parser.parse_args()

config_file_path = args.config_file_path

print(f"Reading Config File {config_file_path}")
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)
    
print("Defining Variables and Creating Directories")

csv_file = config['data']['input_csv']
sqlite_file = config['data']['output_sqlite']
tag = config['base']['tag']
full_column_names_file = config['data']['column_names_file']
column_descriptions_file = config['data']['column_descriptions_file']

sqlite_file = os.path.join(f'outputs/{tag}/data/{sqlite_file}')

print(create_directory_if_not_exists(os.path.join(f'outputs/{tag}/data/')))

print("Initializing Logger")

logger = Logger(config)

logger.log("----------------------Logger Initialized for csv_to_sqlite.py----------------------")

logger.log("Reading Columns Full Names")

with open(full_column_names_file, 'r') as file:
    column_full_names = json.load(file)
    
logger.log("Reading Columns Descriptions")

with open(column_descriptions_file, 'r') as file:
    column_descriptions = json.load(file)

# Loading the loan data file
logger.log('Loading data CSV')
df = pd.read_csv(csv_file, low_memory=False)  # Load the CSV file into a pandas DataFrame

# Extracting metadata from the last two rows of the dataset
logger.log('Loading Metadata')
metadata = df.iloc[-2:]['id'].values  # Store the last two rows' 'id' values as metadata
df = df.iloc[:-2]  # Remove the last two rows from the DataFrame

# Converting date columns to Unix time format
logger.log('Creating UNIX columns for dates')

dates_columns = ['issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line',
                 'sec_app_earliest_cr_line', 'hardship_start_date', 'hardship_end_date',
                 'payment_plan_start_date', 'settlement_date', 'debt_settlement_flag_date']
for column in dates_columns:
    # Convert each date column to Unix time, handling NaN values appropriately
    df[f'{column}_unix'] = df[column].astype('str').apply(lambda x: convert_to_unix_time(x) if x != 'nan' else pd.NA)

# Creating new derived columns for analysis
logger.log('Creating other interesting columns')
df['term_months'] = df['term'].astype('str').apply(lambda x: int(x[1:3]) if x != 'nan' else pd.NA)  # Convert loan term to integer months
# Convert employment length to integer years, handling special cases and NaNs
df['emp_length_years'] = df['emp_length'].astype('str').apply(lambda x: int(x.split(' ')[0].replace('+','')) if x != '< 1 year' and x != 'nan' else pd.NA)
df['loan_id'] = df['url'].astype('str').apply(lambda x: int(x.split('loan_id=')[-1]) if x != 'nan' else pd.NA)  # Extract loan ID from URL
df['id'] = range(len(df))  # Assign a new sequential ID to each row

# Dropping original columns that are no longer needed or have been transformed
logger.log("Dropping proper columns")
drop_columns = ['member_id', 'term', 'emp_length'] + dates_columns
for column in drop_columns:
    df.drop(column, axis='columns', inplace=True)

# Filling missing values in specific columns with 0
logger.log("Filling proper columns")
fillna_columns = ['tot_coll_amt', 'tot_cur_bal']
for column in fillna_columns:
    df[column].fillna(0., inplace=True)

# Casting columns to their appropriate data types
logger.log("Casting columns in proper types")
# Convert specified columns to string, handling NaN values
str_columns = ['emp_title', 'desc', 'title', 'verification_status_joint', 'hardship_type',
               'hardship_reason', 'hardship_status', 'hardship_loan_status', 'settlement_status']
for column in str_columns:
    df[column] = df[column].astype('str').apply(lambda x: x if x != 'nan' else pd.NA)

# Convert specified columns to integer
int_columns = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'annual_inc', 'delinq_2yrs',
               'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec',
               'revol_bal', 'total_acc', 'last_fico_range_low', 'last_fico_range_high',
               'collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', 'tot_coll_amt',
               'tot_cur_bal', 'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens', 'mort_acc', 
               'pub_rec_bankruptcies']

for column in int_columns:
    df[column] = df[column].replace([np.inf, -np.inf], np.nan)
    df[column] = df[column].fillna(-1)
    df[column] = df[column].astype(int)
    df[column] = df[column].astype('Int64')
    df[column] = df[column].replace(-1, pd.NA)

# Convert specified columns to float
float_columns = ['int_rate', 'installment', 'dti', 'revol_util']
for column in float_columns:
    df[column] = df[column].astype('float')

logger.log("Adjusting all missing values to pd.NA")
for column in df.columns:
    df[column] = df[column].replace([None, np.nan], pd.NA)
# Converting metadata to its own dataframe
metadata = pd.DataFrame({s.split(': ')[0]:[int(s.split(': ')[-1])] for s in metadata})

column_types = {}
for column in df.columns:
    col_type = str(df[column].dtype)
    if col_type == 'int64' or column in int_columns:
        column_types[column] = 'INT'
    elif col_type == 'float64' or column in float_columns:
        column_types[column] = 'REAL'
    else:
        column_types[column] = 'TEXT' 

df['emp_title'] = df['emp_title'].astype('str').apply(lambda x: x.lower().strip().replace(',','-').replace('  ',' '))

table_descriptions = {
    'loans_data':f'Data for all the loans available in the database, downloaded as a csv, cleaned, and put into a sqlite file on 2023-12-06. Original download from https://www.kaggle.com/code/pavlofesenko/minimizing-risks-for-loan-investments. Contains columns: {", ".join(df.columns)}.',
    'metadata':f'Metadata provided on the downloaded data, providing the total amount funded in different policy codes. Contains columns: {", ".join(metadata.columns)}.',
    'descriptions':f'A table containing a written description of each available table, and column. Contains columns: loans_data, metadata, descriptions, {", ".join(df.columns)}, {", ".join(metadata.columns)}.'
}

metadata_columns_description = column_descriptions['metadata']

loans_data_columns_description = column_descriptions['loans_data']

descriptions_columns_description = column_descriptions['descriptions']

descriptions = {'name':[], 'full_name':[], 'type':[], 'location':[], 'description':[], 'data_type':[]}
    
for key, value in table_descriptions.items():
    descriptions['name'] += [key]
    descriptions['full_name'] += [key]
    descriptions['type'] += ['table']
    descriptions['location'] += ['root']
    descriptions['description'] += [value]
    descriptions['data_type'] += ['TABLE']
    
for key, value in metadata_columns_description.items():
    descriptions['name'] += [key]
    descriptions['full_name'] += [key]
    descriptions['type'] += ['column']
    descriptions['location'] += ['metadata']
    descriptions['description'] += [value]
    descriptions['data_type'] += ['INT']

for key, value in loans_data_columns_description.items():
    descriptions['name'] += [key]
    descriptions['full_name'] += [column_full_names[key]]
    descriptions['type'] += ['column']
    descriptions['location'] += ['loans_data']
    descriptions['description'] += [value]
    descriptions['data_type'] += [column_types[key]]
    
for key, value in descriptions_columns_description.items():
    descriptions['name'] += [key]
    descriptions['full_name'] += [key]
    descriptions['type'] += ['column']
    descriptions['location'] += ['descriptions']
    descriptions['description'] += [value]
    descriptions['data_type'] += ['TEXT']

descriptions = pd.DataFrame(descriptions)

# Create a SQLite database
logger.log("Connecting to Database")
conn = sqlite3.connect(sqlite_file)

logger.log("Creating Queries")
# Query to delete loans_data if it exists
drop_loans_data_query = 'DROP TABLE IF EXISTS loans_data'
# Query to delete metadata if it exists
drop_metadata_query = 'DROP TABLE IF EXISTS metadata'
# Query to delete descriptions if it exists
drop_descriptions_query = 'DROP TABLE IF EXISTS descriptions'

# Query to create a table for the loans data
create_loans_data_table_query = 'CREATE TABLE loans_data (' + ', '.join([f"\"{col}\" {col_type}" for col, col_type in column_types.items()]) + ')'
# Query to create a table for the metadata
create_metadata_table_query = 'CREATE TABLE metadata (' + ', '.join([f"\"{col}\" TEXT" for col in metadata.columns]) + ')'
# Query to create a table for the descriptions
create_descriptions_table_query = 'CREATE TABLE descriptions (' + ', '.join([f"\"{col}\" TEXT" for col in descriptions.columns]) + ')'

logger.log("Dropping old tables and creating new ones")
# Drops and creates the tables
conn.execute(drop_loans_data_query)
conn.execute(create_loans_data_table_query)
conn.execute(drop_metadata_query)
conn.execute(create_metadata_table_query)
conn.execute(drop_descriptions_query)
conn.execute(create_descriptions_table_query)

logger.log("Loading data into tables")
# Insert data from DataFrame to the SQLite table
df.to_sql('loans_data', conn, if_exists='replace', index=False)
metadata.to_sql('metadata', conn, if_exists='replace', index=False)
descriptions.to_sql('descriptions', conn, if_exists='replace', index=False)

conn.close()
logger.log("----------------------Done with csv_to_sqlite.py----------------------")
