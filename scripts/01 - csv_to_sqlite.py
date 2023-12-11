"""
This script processes loan data from a CSV file and converts it into an SQLite database.

The script reads a CSV file containing loan data, performs data cleaning operations such as 
dropping rows with missing values in specific columns, and converts certain date columns 
to Unix time format. The cleaned data is then saved into a specified SQLite database file.

Command-line arguments:
  --input_csv (str): The file path to the input CSV file containing loan data.
                     Example usage: --input_csv 'path/to/input.csv'
  --output_sqlite (str): The file path for the output SQLite database file where the processed data will be stored.
                         Example usage: --output_sqlite 'path/to/output.sqlite'

The script expects the input CSV file to be in a specific format with columns like 'term', 'annual_inc', etc.
It outputs an SQLite database with the cleaned and processed data.

Usage:
  To run the script, provide the paths to the input CSV and the output SQLite file. For example:
  python "01 - csv_to_sqlite.py" --input_csv 'path/to/input.csv' --output_sqlite 'path/to/output.sqlite'

Dependencies:
  pandas: Used for data manipulation and analysis.
  sqlite3: Used for SQLite database management.
  numpy, datetime, time: Used for various data processing tasks.

Author: [Your Name]
Date: [Date of Script Creation]
"""
# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import sqlite3       # For SQLite database management
import numpy as np   # For numerical operations
import datetime      # For handling date and time
import time          # For time-related tasks
import argparse # Import argparse for command-line argument parsing

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
        
# Formal name of each column
column_full_names = {
    'id': 'ID',
    'loan_amnt': 'Loan Amount (USD)',
    'funded_amnt': 'Funded Amount (USD)',
    'funded_amnt_inv': 'Funded Amount by Investors (USD)',
    'int_rate': 'Interest Rate (%)',
    'installment': 'Installment (USD)',
    'grade': 'Loan Grade',
    'sub_grade': 'Loan Subgrade',
    'emp_title': 'Employee Title',
    'home_ownership': 'Home Ownership Status',
    'annual_inc': 'Annual Income (USD)',
    'verification_status': 'Income Verification Status',
    'loan_status': 'Loan Status',
    'pymnt_plan': 'Payment Plan',
    'url': 'Loan URL',
    'desc': 'Loan Description',
    'purpose': 'Loan Purpose',
    'title': 'Loan Title',
    'zip_code': 'Zip Code',
    'addr_state': 'State Address',
    'dti': 'Debt-to-Income Ratio (%)',
    'delinq_2yrs': 'Number of 30+ days Delinquency in Past 2 Years',
    'earliest_cr_line': 'Date of Earliest Credit Line',
    'earliest_cr_line_unix': 'Date of Earliest Credit Line (UNIX Timestamp)',
    'fico_range_low': 'FICO Score Range Low',
    'fico_range_high': 'FICO Score Range High',
    'inq_last_6mths': 'Inquiries in Last 6 Months',
    'mths_since_last_delinq': 'Months Since Last Delinquency',
    'mths_since_last_record': 'Months Since Last Public Record',
    'open_acc': 'Number of Open Credit Lines',
    'pub_rec': 'Number of Public Records',
    'revol_bal': 'Revolving Balance (USD)',
    'revol_util': 'Revolving Line Utilization Rate (%)',
    'total_acc': 'Total Number of Credit Lines',
    'initial_list_status': 'Initial Listing Status',
    'out_prncp': 'Outstanding Principal (USD)',
    'out_prncp_inv': 'Outstanding Principal by Investors (USD)',
    'total_pymnt': 'Total Payment Received (USD)',
    'total_pymnt_inv': 'Total Payment Received by Investors (USD)',
    'total_rec_prncp': 'Total Principal Received (USD)',
    'total_rec_int': 'Total Interest Received (USD)',
    'total_rec_late_fee': 'Total Late Fees Received (USD)',
    'recoveries': 'Recoveries (USD)',
    'collection_recovery_fee': 'Collection Recovery Fee (USD)',
    'last_pymnt_amnt': 'Last Payment Amount (USD)',
    'last_fico_range_high': 'Last FICO Score Range High',
    'last_fico_range_low': 'Last FICO Score Range Low',
    'collections_12_mths_ex_med': 'Collections Excluding Medical in Past 12 Months',
    'mths_since_last_major_derog': 'Months Since Last Major Derogatory',
    'policy_code': 'Policy Code',
    'application_type': 'Application Type',
    'annual_inc_joint': 'Joint Annual Income (USD)',
    'dti_joint': 'Joint Debt-to-Income Ratio (%)',
    'verification_status_joint': 'Joint Income Verification Status',
    'acc_now_delinq': 'Accounts Currently Delinquent',
    'tot_coll_amt': 'Total Collection Amount (USD)',
    'tot_cur_bal': 'Total Current Balance (USD)',
    'open_acc_6m': 'Open Accounts in Last 6 Months',
    'open_act_il': 'Open Installment Loans',
    'open_il_12m': 'Open Installment Loans in Last 12 Months',
    'open_il_24m': 'Open Installment Loans in Last 24 Months',
    'mths_since_rcnt_il': 'Months Since Most Recent Installment Loan',
    'total_bal_il': 'Total Balance on Installment Loans (USD)',
    'il_util': 'Installment Loan Utilization Rate (%)',
    'open_rv_12m': 'Open Revolving Accounts in Last 12 Months',
    'open_rv_24m': 'Open Revolving Accounts in Last 24 Months',
    'max_bal_bc': 'Maximum Balance on Bankcard (USD)',
    'all_util': 'Total Utilization Rate (%)',
    'total_rev_hi_lim': 'Total Revolving High Credit/Loan Limit (USD)',
    'inq_fi': 'Inquiries from Financial Institutions',
    'total_cu_tl': 'Total Consumer Unit Accounts',
    'inq_last_12m': 'Inquiries in Last 12 Months',
    'acc_open_past_24mths': 'Accounts Opened in Past 24 Months',
    'avg_cur_bal': 'Average Current Balance (USD)',
    'bc_open_to_buy': 'Bankcard Open to Buy (USD)',
    'bc_util': 'Bankcard Utilization Rate (%)',
    'chargeoff_within_12_mths': 'Charge-offs Within 12 Months',
    'delinq_amnt': 'Delinquent Amount (USD)',
    'mo_sin_old_il_acct': 'Months Since Oldest Installment Account',
    'mo_sin_old_rev_tl_op': 'Months Since Oldest Revolving Account',
    'mo_sin_rcnt_rev_tl_op': 'Months Since Most Recent Revolving Account',
    'mo_sin_rcnt_tl': 'Months Since Most Recent Account',
    'mort_acc': 'Number of Mortgage Accounts',
    'mths_since_recent_bc': 'Months Since Most Recent Bankcard Account',
    'mths_since_recent_bc_dlq': 'Months Since Recent Bankcard Delinquency',
    'mths_since_recent_inq': 'Months Since Recent Inquiry',
    'mths_since_recent_revol_delinq': 'Months Since Recent Revolving Delinquency',
    'num_accts_ever_120_pd': 'Number of Accounts Ever 120 Days Past Due',
    'num_actv_bc_tl': 'Number of Active Bankcard Accounts',
    'num_actv_rev_tl': 'Number of Active Revolving Trades',
    'num_bc_sats': 'Number of Satisfactory Bankcard Accounts',
    'num_bc_tl': 'Number of Bankcard Accounts',
    'num_il_tl': 'Number of Installment Loans',
    'num_op_rev_tl': 'Number of Open Revolving Accounts',
    'num_rev_accts': 'Number of Revolving Accounts',
    'num_rev_tl_bal_gt_0': 'Number of Revolving Trades with Balance Greater than 0',
    'num_sats': 'Number of Satisfactory Accounts',
    'num_tl_120dpd_2m': 'Number of Accounts 120 Days Past Due in Last 2 Months',
    'num_tl_30dpd': 'Number of Accounts 30 Days Past Due',
    'num_tl_90g_dpd_24m': 'Number of Accounts 90+ Days Past Due in Last 24 Months',
    'num_tl_op_past_12m': 'Number of Accounts Opened in Past 12 Months',
    'pct_tl_nvr_dlq': 'Percentage of Trades Never Delinquent (%)',
    'percent_bc_gt_75': 'Percentage of Bankcard Accounts > 75% Utilization (%)',
    'pub_rec_bankruptcies': 'Number of Public Record Bankruptcies',
    'tax_liens': 'Number of Tax Liens',
    'tot_hi_cred_lim': 'Total High Credit Limit (USD)',
    'total_bal_ex_mort': 'Total Balance Excluding Mortgage (USD)',
    'total_bc_limit': 'Total Bankcard Limit (USD)',
    'total_il_high_credit_limit': 'Total Installment High Credit/Limit (USD)',
    'revol_bal_joint': 'Joint Revolving Balance (USD)',
    'sec_app_fico_range_low': 'Secondary Applicant FICO Score Range Low',
    'sec_app_fico_range_high': 'Secondary Applicant FICO Score Range High',
    'sec_app_inq_last_6mths': 'Secondary Applicant Inquiries in Last 6 Months',
    'sec_app_mort_acc': 'Secondary Applicant Mortgage Accounts',
    'sec_app_open_acc': 'Secondary Applicant Open Accounts',
    'sec_app_revol_util': 'Secondary Applicant Revolving Utilization Rate (%)',
    'sec_app_open_act_il': 'Secondary Applicant Open Installment Loans',
    'sec_app_num_rev_accts': 'Secondary Applicant Number of Revolving Accounts',
    'sec_app_chargeoff_within_12_mths': 'Secondary Applicant Charge-offs Within 12 Months',
    'sec_app_collections_12_mths_ex_med': 'Secondary Applicant Collections Excluding Medical in Past 12 Months',
    'sec_app_mths_since_last_major_derog': 'Months Since Last Major Derogatory for Secondary Applicant',
    'hardship_flag': 'Hardship Flag',
    'hardship_type': 'Type of Hardship Plan',
    'hardship_reason': 'Reason for Hardship',
    'hardship_status': 'Status of Hardship Plan',
    'deferral_term': 'Term of Payment Deferral (Months)',
    'hardship_amount': 'Hardship Plan Amount (USD)',
    'hardship_length': 'Length of Hardship Plan (Months)',
    'hardship_dpd': 'Days Past Due under Hardship Plan',
    'hardship_loan_status': 'Loan Status under Hardship Plan',
    'orig_projected_additional_accrued_interest': 'Original Projected Additional Accrued Interest (USD)',
    'hardship_payoff_balance_amount': 'Payoff Balance Amount under Hardship (USD)',
    'hardship_last_payment_amount': 'Last Payment Amount under Hardship (USD)',
    'disbursement_method': 'Disbursement Method',
    'debt_settlement_flag': 'Debt Settlement Flag',
    'settlement_status': 'Status of Debt Settlement',
    'settlement_amount': 'Debt Settlement Amount (USD)',
    'settlement_percentage': 'Debt Settlement Percentage (%)',
    'settlement_term': 'Debt Settlement Term (Months)',
    'issue_d': 'Issue Date',
    'issue_d_unix': 'Issue Date (Unix Timestamp)',
    'last_pymnt_d': 'Last Payment Date',
    'last_pymnt_d_unix': 'Last Payment Date (Unix Timestamp)',
    'next_pymnt_d': 'Next Payment Date',
    'next_pymnt_d_unix': 'Next Payment Date (Unix Timestamp)',
    'last_credit_pull_d': 'Last Credit Pull Date',
    'last_credit_pull_d_unix': 'Last Credit Pull Date (Unix Timestamp)',
    'sec_app_earliest_cr_line': 'Secondary Applicant Earliest Credit Line',
    'sec_app_earliest_cr_line_unix': 'Secondary Applicant Earliest Credit Line (Unix Timestamp)',
    'hardship_start_date': 'Hardship Start Date',
    'hardship_start_date_unix': 'Hardship Start Date (Unix Timestamp)',
    'hardship_end_date': 'Hardship End Date',
    'hardship_end_date_unix': 'Hardship End Date (Unix Timestamp)',
    'payment_plan_start_date': 'Payment Plan Start Date',
    'payment_plan_start_date_unix': 'Payment Plan Start Date (Unix Timestamp)',
    'settlement_date': 'Debt Settlement Date',
    'settlement_date_unix': 'Debt Settlement Date (Unix Timestamp)',
    'debt_settlement_flag_date': 'Date of Debt Settlement Flag',
    'debt_settlement_flag_date_unix': 'Date of Debt Settlement Flag (Unix Timestamp)',
    'term_months': 'Loan Term (Months)',
    'emp_length_years': 'Employment Length (Years)',
    'loan_id': 'Loan ID',}

parser = argparse.ArgumentParser(description='Process CSV file to SQLite database.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output_sqlite', type=str, required=True, help='Path to the output SQLite file')

# Parse the arguments
args = parser.parse_args()

csv_file = args.input_csv  # Input CSV file path from command-line arguments
sqlite_file = args.output_sqlite  # Output SQLite file path from command-line arguments

# Loading the loan data file
print('Loading File')
df = pd.read_csv(csv_file, low_memory=False)  # Load the CSV file into a pandas DataFrame

# Extracting metadata from the last two rows of the dataset
print('Loading Metadata')
metadata = df.iloc[-2:]['id'].values  # Store the last two rows' 'id' values as metadata
df = df.iloc[:-2]  # Remove the last two rows from the DataFrame

# Converting date columns to Unix time format
print('Creating UNIX columns for dates')
dates_columns = ['issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line',
                 'sec_app_earliest_cr_line', 'hardship_start_date', 'hardship_end_date',
                 'payment_plan_start_date', 'settlement_date', 'debt_settlement_flag_date']
for column in dates_columns:
    # Convert each date column to Unix time, handling NaN values appropriately
    df[f'{column}_unix'] = df[column].astype('str').apply(lambda x: convert_to_unix_time(x) if x != 'nan' else pd.NA)

# Creating new derived columns for analysis
print('Creating other interesting columns')
df['term_months'] = df['term'].astype('str').apply(lambda x: int(x[1:3]) if x != 'nan' else pd.NA)  # Convert loan term to integer months
# Convert employment length to integer years, handling special cases and NaNs
df['emp_length_years'] = df['emp_length'].astype('str').apply(lambda x: int(x.split(' ')[0].replace('+','')) if x != '< 1 year' and x != 'nan' else pd.NA)
df['loan_id'] = df['url'].astype('str').apply(lambda x: int(x.split('loan_id=')[-1]) if x != 'nan' else pd.NA)  # Extract loan ID from URL
df['id'] = range(len(df))  # Assign a new sequential ID to each row

# Dropping original columns that are no longer needed or have been transformed
print("Dropping proper columns")
drop_columns = ['member_id', 'term', 'emp_length'] + dates_columns
for column in drop_columns:
    df.drop(column, axis='columns', inplace=True)

# Filling missing values in specific columns with 0
print("Filling proper columns")
fillna_columns = ['tot_coll_amt', 'tot_cur_bal']
for column in fillna_columns:
    df[column].fillna(0., inplace=True)

# Casting columns to their appropriate data types
print("Casting columns in proper types")
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

print("Adjusting all missing values to pd.NA")
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

metadata_columns_description = {
    'Total amount funded in policy code 1': 'Contains the total amount, in dollars, funded in loans with policy code 1.',
    'Total amount funded in policy code 2': 'Contains the total amount, in dollars, funded in loans with policy code 2.'
}

loans_data_columns_description = {
    'id':'A row identifyer for the loans table. Each id is unique for each row.',
    'loan_amnt': 'The listed amount of the loan applied for by the borrower.',
    'funded_amnt': 'The total amount committed to the loan by the Lending Club.',
    'funded_amnt_inv': 'The total amount committed by investors for the loan.',
    'int_rate': 'Interest rate of the loan.',
    'installment': 'The monthly payment owed by the borrower if the loan originates.',
    'grade': 'Lending Club assigned loan grade.',
    'sub_grade': 'Lending Club assigned sub-grade.',
    'emp_title': 'The job title supplied by the borrower when applying for the loan.',
    'home_ownership': 'The home ownership status provided by the borrower during registration. Values are: RENT, OWN, MORTGAGE, OTHER.',
    'annual_inc': 'The self-reported annual income provided by the borrower during registration.',
    'verification_status': 'Indicates if income was verified by Lending Club.',
    'loan_status': 'Current status of the loan.',
    'pymnt_plan': 'Indicates if a payment plan has been put in place for the loan.',
    'url': 'URL for the Lending Club page with detailed information about the loan.',
    'desc': 'Loan description provided by the borrower.',
    'purpose': 'A category provided by the borrower for the loan request.',
    'title': 'The loan title provided by the borrower.',
    'zip_code': 'The first 3 numbers of the zip code provided by the borrower in the loan application.',
    'addr_state': 'The state provided by the borrower in the loan application.',
    'dti': 'Debt-to-income ratio calculated using the borrower’s total monthly debt payments divided by their self-reported monthly income.',
    'delinq_2yrs': 'The number of 30+ days delinquencies the borrower has had in the past 2 years.',
    'earliest_cr_line': 'The month and year the borrower’s earliest reported credit line was opened.',
    'fico_range_low': 'The lower boundary range of the borrower’s FICO score at the time of loan origination.',
    'fico_range_high': 'The upper boundary range of the borrower’s FICO score at the time of loan origination.',
    'inq_last_6mths': 'The number of credit inquiries the borrower has had in the last 6 months (excluding auto and mortgage inquiries).',
    'mths_since_last_delinq': 'The number of months since the borrower’s last delinquency.',
    'mths_since_last_record': 'The number of months since the last public record.',
    'open_acc': 'The number of open credit lines in the borrower’s credit file.',
    'pub_rec': 'Number of derogatory public records, such as bankruptcy filings, tax liens, or judgments.',
    'revol_bal': 'The total amount of credit revolving balances.',
    'revol_util': 'Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.',
    'total_acc': 'The total number of credit lines currently in the borrower’s credit file.',
    'initial_list_status': 'The initial listing status of the loan. Possible values are – W, F.',
    'out_prncp': 'The remaining outstanding principal for total amount funded.',
    'out_prncp_inv': 'The remaining outstanding principal for portion of total amount funded by investors.',
    'total_pymnt': 'Payments received to date for total amount funded.',
    'total_pymnt_inv': 'Payments received to date for portion of total amount funded by investors.',
    'total_rec_prncp': 'Principal received to date.',
    'total_rec_int': 'Interest received to date.',
    'total_rec_late_fee': 'Late fees received to date.',
    'recoveries': 'Post charge off gross recovery.',
    'collection_recovery_fee': 'Post charge off collection fee.',
    'last_pymnt_amnt': 'Last total payment amount received.',
    'last_fico_range_high': 'The upper boundary range of the borrower’s last FICO score.',
    'last_fico_range_low': 'The lower boundary range of the borrower’s last FICO score.',
    'collections_12_mths_ex_med': 'Number of collections in 12 months excluding medical collections.',
    'mths_since_last_major_derog': 'Months since most recent 90-day or worse rating.',
    'policy_code': 'Publicly available policy_code=1, new products not publicly available policy_code=2.',
    'application_type': 'Indicates whether the loan is an individual application or a joint application with two co-borrowers.',
    'annual_inc_joint': 'The combined annual income reported by the co-borrowers on a joint application.',
    'dti_joint': 'The debt-to-income ratio calculated using the total monthly debt payments of the co-borrowers, divided by the combined self-reported monthly income of the co-borrowers, on a joint application.',
    'verification_status_joint': 'Indicates if the co-borrowers’ joint income was verified by Lending Club.',
    'acc_now_delinq': 'The number of accounts on which the borrower is now delinquent.',
    'tot_coll_amt': 'Total collection amounts ever owed.',
    'tot_cur_bal': 'Total current balance of all accounts.',
    'open_acc_6m': 'Number of open trades in the last 6 months.',
    'open_act_il': 'Number of currently active installment trades.',
    'open_il_12m': 'Number of installment accounts opened in past 12 months.',
    'open_il_24m': 'Number of installment accounts opened in past 24 months.',
    'mths_since_rcnt_il': 'Months since the most recent installment account opened.',
    'total_bal_il': 'Total current balance of all installment accounts.',
    'il_util': 'Ratio of total current balance to high credit/credit limit on all installment accounts.',
    'open_rv_12m': 'Number of revolving trades opened in past 12 months.',
    'open_rv_24m': 'Number of revolving trades opened in past 24 months.',
    'max_bal_bc': 'Maximum current balance owed on all revolving accounts.',
    'all_util': 'Balance to credit limit on all trades.',
    'total_rev_hi_lim': 'Total revolving high credit/credit limit.',
    'inq_fi': 'Number of personal finance inquiries.',
    'total_cu_tl': 'Number of finance trades.',
    'inq_last_12m': 'Number of credit inquiries in the past 12 months.',
    'acc_open_past_24mths': 'Number of accounts opened in past 24 months.',
    'avg_cur_bal': 'Average current balance of all accounts.',
    'bc_open_to_buy': 'Total open to buy on revolving bankcards.',
    'bc_util': 'Ratio of total current balance to high credit/credit limit for all bankcard accounts.',
    'chargeoff_within_12_mths': 'Number of charge-offs within  last 12 months.',
    'delinq_amnt': 'The past-due amount owed for the accounts on which the borrower is now delinquent.',
    'mo_sin_old_il_acct': 'Months since oldest bank installment account opened.',
    'mo_sin_old_rev_tl_op': 'Months since oldest revolving account opened.',
    'mo_sin_rcnt_rev_tl_op': 'Months since most recent revolving account opened.',
    'mo_sin_rcnt_tl': 'Months since most recent account opened.',
    'mort_acc': 'Number of mortgage accounts.',
    'mths_since_recent_bc': 'Months since most recent bankcard account opened.',
    'mths_since_recent_bc_dlq': 'Months since most recent bankcard delinquency.',
    'mths_since_recent_inq': 'Months since most recent inquiry.',
    'mths_since_recent_revol_delinq': 'Months since most recent revolving delinquency.',
    'num_accts_ever_120_pd': 'Number of accounts ever 120 or more days past due.',
    'num_actv_bc_tl': 'Number of currently active bankcard accounts.',
    'num_actv_rev_tl': 'Number of currently active revolving trades.',
    'num_bc_sats': 'Number of satisfactory bankcard accounts.',
    'num_bc_tl': 'Number of bankcard accounts (credit cards issued by banks).',
    'num_il_tl': 'Number of installment loan accounts (loans with fixed payments and a set maturity date).',
    'num_op_rev_tl': 'Number of open revolving accounts (credit cards and lines of credit).',
    'num_rev_accts': 'Number of revolving accounts.',
    'num_rev_tl_bal_gt_0': 'Number of revolving trades with balance greater than zero.',
    'num_sats': 'Number of satisfactory accounts.',
    'num_tl_120dpd_2m': 'Number of accounts currently 120 days past due (updated in past 2 months).',
    'num_tl_30dpd': 'Number of accounts currently 30 days past due.',
    'num_tl_90g_dpd_24m': 'Number of accounts 90 or more days past due in last 24 months.',
    'num_tl_op_past_12m': 'Number of accounts opened in past 12 months.',
    'pct_tl_nvr_dlq': 'Percentage of accounts never delinquent.',
    'percent_bc_gt_75': 'Percentage of bankcard accounts with credit utilization greater than 75%.',
    'pub_rec_bankruptcies': 'Number of public record bankruptcies.',
    'tax_liens': 'Number of tax liens.',
    'tot_hi_cred_lim': 'Total high credit/credit limit (the total of the maximum amount of credit extended on all accounts).',
    'total_bal_ex_mort': 'Total credit balance excluding mortgage.',
    'total_bc_limit': 'Total bankcard (credit card) limit.',
    'total_il_high_credit_limit': 'Total installment high credit/credit limit.',
    'revol_bal_joint': 'Total revolving balance on joint accounts.',
    'sec_app_fico_range_low': 'FICO range (low) for the secondary applicant.',
    'sec_app_fico_range_high': 'FICO range (high) for the secondary applicant.',
    'sec_app_inq_last_6mths': 'Number of credit inquiries for the secondary applicant in the last 6 months.',
    'sec_app_mort_acc': 'Number of mortgage accounts held by the secondary applicant.',
    'sec_app_open_acc': 'Number of open credit lines in the secondary applicant’s credit file.',
    'sec_app_revol_util': 'Revolving line utilization rate, or the amount of credit the secondary applicant is using relative to all available revolving credit.',
    'sec_app_open_act_il': 'Number of currently active installment trades for the secondary applicant.',
    'sec_app_num_rev_accts': 'Number of revolving accounts held by the secondary applicant.',
    'sec_app_chargeoff_within_12_mths': 'Number of charge-offs within the last 12 months for the secondary applicant.',
    'sec_app_collections_12_mths_ex_med': 'Number of collections in 12 months excluding medical collections for the secondary applicant.',
    'sec_app_mths_since_last_major_derog': 'Months since the secondary applicant’s last major derogatory event.',
    'hardship_flag': 'Flags whether the borrower is on a hardship plan.',
    'hardship_type': 'Type of hardship plan offered to the borrower, if any.',
    'hardship_reason': 'Reason provided by the borrower for the hardship event.',
    'hardship_status': 'Current status of the hardship plan.',
    'deferral_term': 'The number of months that the borrower is expected to pay less than the contractual monthly payment amount due to a hardship plan.',
    'hardship_amount': 'The amount of the monthly payment that the borrower must pay while under a hardship plan.',
    'hardship_length': 'The length of the hardship plan in months.',
    'hardship_dpd': 'Number of days past due as of the hardship plan start date.',
    'hardship_loan_status': 'Loan status as of the hardship plan start date.',
    'orig_projected_additional_accrued_interest': 'The original estimated total amount of additional interest the borrower will pay as a result of the hardship plan.',
    'hardship_payoff_balance_amount': 'The balance amount at the time the borrower entered into a hardship plan.',
    'hardship_last_payment_amount': 'The last payment amount made by the borrower before entering the hardship plan.',
    'disbursement_method': 'Method by which the borrower receives their loan. Possible values are \'cash\', \'directpay\', etc.',
    'debt_settlement_flag': 'Indicates whether the borrower, who has charged-off, is working with a debt settlement company.',
    'settlement_status': 'The status of the borrower’s settlement plan. Possible values are \'active\', \'completed\', \'broken\', etc.',
    'settlement_amount': 'The amount that the borrower has agreed to pay in a settlement plan.',
    'settlement_percentage': 'The percentage of the unpaid principal balance the borrower has agreed to pay in a settlement plan.',
    'settlement_term': 'The number of months the borrower will be on the settlement plan.',
    'issue_d_unix': 'The date when the loan was funded, converted to Unix time.',
    'last_pymnt_d_unix': 'The date of the last payment received, converted to Unix time.',
    'next_pymnt_d_unix': 'The date of the next scheduled payment, converted to Unix time.',
    'last_credit_pull_d_unix': 'The most recent date Lending Club pulled credit for this loan, converted to Unix time.',
    'sec_app_earliest_cr_line_unix': 'Earliest credit line date for the secondary applicant, converted to Unix time.',
    'hardship_start_date_unix': 'The start date of the hardship plan, converted to Unix time.',
    'hardship_end_date_unix': 'The end date of the hardship plan, converted to Unix time.',
    'payment_plan_start_date_unix': 'The start date of the payment plan, converted to Unix time.',
    'settlement_date_unix': 'The date that the borrower agrees to the settlement plan, converted to Unix time.',
    'debt_settlement_flag_date_unix': 'The date that the borrower\'s debt settlement flag was set, converted to Unix time.',
    'term_months': 'The length of the loan term in months.',
    'emp_length_years': 'The borrower’s length of employment in years.',
    'loan_id':'Identification nunber of the loan according to the url associated',
}

descriptions_columns_description = {
    'name': 'The name of the element to be described',
    'type': 'The type of the element, table or column',
    'location': 'The table where the element is located (root if it\'s a table)',
    'description': 'A description of the element'
}

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
print("Connecting to Database")
conn = sqlite3.connect(sqlite_file)

print("Creating Queries")
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

print("Dropping old tables and creating new ones")
# Drops and creates the tables
conn.execute(drop_loans_data_query)
conn.execute(create_loans_data_table_query)
conn.execute(drop_metadata_query)
conn.execute(create_metadata_table_query)
conn.execute(drop_descriptions_query)
conn.execute(create_descriptions_table_query)

print("Loading data into tables")
# Insert data from DataFrame to the SQLite table
df.to_sql('loans_data', conn, if_exists='replace', index=False)
metadata.to_sql('metadata', conn, if_exists='replace', index=False)
descriptions.to_sql('descriptions', conn, if_exists='replace', index=False)

conn.close()
print("Done")