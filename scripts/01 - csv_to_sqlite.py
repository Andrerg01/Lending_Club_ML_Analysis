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

def cast_proper_type(column):
    # Check for boolean conversion (including special case for 'Y' and 'N')
    if all(val in [0, 1, '0', '1', 'Y', 'N', True, False] for val in column):
        return column.replace({'Y': True, 'N': False}).astype(bool)

    # Check for integer conversion
    if all(pd.to_numeric(column, errors='coerce').notnull()) and (pd.to_numeric(column).dropna() % 1 == 0).all():
        return pd.to_numeric(column, downcast='integer')

    # Check for float conversion
    if all(pd.to_numeric(column, errors='coerce').notnull()):
        return pd.to_numeric(column, downcast='float')

    # Default to string conversion
    return column.astype(str)
        
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
df = pd.read_csv(csv_file, low_memory=False).replace('nan',None)  # Load the CSV file into a pandas DataFrame

# Extracting metadata from the last two rows of the dataset
logger.log('Loading Metadata')
metadata = df.iloc[-2:]['id'].values  # Store the last two rows' 'id' values as metadata
df = df.iloc[:-2]  # Remove the last two rows from the DataFrame

# Converting date columns to Unix time format
logger.log('Finding Date Columns')

dates_columns = []
for column in sorted(df.columns):
    samples = df[column].dropna().unique()
    if len(samples) > 0 and len(str(samples[0])) == 8 and str(samples[0])[3] == '-':
        dates_columns += [column]

logger.log('Creating UNIX Timestamp Columns for Dates')

for column in dates_columns:
    logger.log(f"Converting column {column} to UNIX time")

    df[f'{column}_dt'] = pd.to_datetime(df[column], format='%b-%Y', errors='coerce')

    df[f'{column}_dt'].fillna(pd.Timestamp('1970-01-01'), inplace=True)

    df[f'{column}_unix'] = (df[f'{column}_dt'] - pd.Timestamp('1970-01-01')).dt.total_seconds().astype(int)

logger.log('Creating interesting columns')

df['term_months'] = df['term'].astype('str').apply(lambda x: int(x[1:3]) if x != 'nan' else -1)

df['emp_length_years'] = df['emp_length'].astype('str').apply(lambda x: int(x.split(' ')[0].replace('+','')) if x != '< 1 year' and x != 'nan' else -1)

df['id'] = range(len(df))

logger.log("Dropping uninteresting columns")

drop_columns = ['member_id', 'term', 'emp_length'] + \
                dates_columns + \
                [f'{col}_dt' for col in dates_columns]

for column in drop_columns:
    logger.log(f'Dropping column {column}')
    if column in df.columns:
        df.drop(column, axis='columns', inplace=True)

logger.log("Filling NaNs on some columns")

fillna_neg_one_columns = ['tot_coll_amt', 'tot_cur_bal', 'all_util', 'annual_inc_joint', 'bc_open_to_buy', 'deferral_term', 'collection_recovery_fee', 'hardship_last_payment_amount', 'hardship_payoff_balance_amount', 'max_bal_bc', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'revol_bal_joint', 'sec_app_fico_range_high', 'sec_app_fico_range_low', 'sec_app_mort_acc',  'sec_app_mths_since_last_major_derog', 'settlement_amount', 'settlement_percentage', 'settlement_term', 'zip_code', 'total_rev_hi_lim', 'tot_hi_cred_lim', 'total_bc_limit', 'total_il_high_credit_limit']

for column in fillna_neg_one_columns:
    logger.log(f'Filling NaNs in column {column} with -1')
    df[column].fillna(-1, inplace=True)
    
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
for column in fillna_zero_columns:
    logger.log(f'Filling NaNs in column {column} with 0')
    df[column].fillna(0, inplace=True)
    
fillna_one_columns = ['bc_util', 'dti', 'dti_joint', 'il_util', 'revol_util', 'sec_app_revol_util']
for column in fillna_one_columns:
    logger.log(f'Filling NaNs in column {column} with 1')
    df[column].fillna(1, inplace=True)
    
fillna_N_columns = ['debt_settlement_flag', 'hardship_flag']
for column in fillna_N_columns:
    logger.log(f'Filling NaNs in column {column} with \'N\'')
    df[column].fillna('N', inplace=True)
    
fillna_empty_string_columns = ['desc', 'emp_title', 'addr_state', 'application_type', 'disbursement_method',
                              'hardship_loan_status', 'hardship_reason', 'hardship_status', 'settlement_status',
                              'title', 'verification_status_joint', 'hardship_type']
for column in fillna_empty_string_columns:
    logger.log(f'Filling NaNs in column {column} with an empty string')
    df[column].fillna('', inplace=True)
    
fillna_100_columns = ['pct_tl_nvr_dlq', 'percent_bc_gt_75']

for column in fillna_100_columns:
    logger.log(f'Filling NaNs in column {column} with 100')
    df[column].fillna(100, inplace=True)
    

dropna_columns = ['fico_range_high', 'fico_range_low', 'funded_amnt', 'funded_amnt_inv', 'grade']
logger.log(f'Dropping columns: {dropna_columns}')
df.dropna(subset=dropna_columns, axis=0, inplace=True)   

logger.log("Modifying Columns")

df['emp_title'] = df['emp_title'].astype('str').apply(lambda x: x.lower().strip().replace(',','-').replace('  ',' '))

df['zip_code'] = df['zip_code'].astype('str').apply(lambda x: x[:3] if x != '' else x)

df['zip_code'] = df['zip_code'].str.lstrip('0')

logger.log("Casting columns to Best Types")

for column in sorted(df.columns):
    logger.log(f'Converting {column} elements from {str(df[column].dtype)} to {str(cast_proper_type(df[column]).dtype)}')
          
    df[column] = cast_proper_type(df[column])

logger.log("Creating Metadata Dataframe")

metadata = pd.DataFrame({s.split(': ')[0]:[int(s.split(': ')[-1])] for s in metadata})

logger.log("Creating Descriptions Dataframe")

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
    descriptions['data_type'] += ['INTEGER']

for key, value in loans_data_columns_description.items():
    if key in df.columns:
        descriptions['name'] += [key]
        descriptions['full_name'] += [column_full_names[key]]
        descriptions['type'] += ['column']
        descriptions['location'] += ['loans_data']
        descriptions['description'] += [value]
        descriptions['data_type'] += [map_dtype_to_sqlite(str(df[key].dtype))]
    
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
