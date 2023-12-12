print("Starting Up")
import pandas as pd
import sqlite3
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import yaml
import argparse

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
    
def transform_type(TYPE):
    if TYPE == 'INT':
        return 'int64'
    elif TYPE == 'REAL':
        return 'float64'
    else:
        return 'object'
    
def loan_status_to_int(status):
    if status == 'Charged Off':
        return 0
    if status == 'Fully Paid':
        return 1
    else:
        return -1

def bestbandwidth(data):
    return 1.06*np.std(data)*len(data)**(-1/5)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return f"Created directory: {directory}"
    else:
        return f"Directory already exists: {directory}"

print("Reading Command Line Arguments")

parser = argparse.ArgumentParser(description="Read configuration file.")

parser.add_argument("--config", required=True, help="Path to the config file", dest="config_file_path")

args = parser.parse_args()

config_file_path = args.config_file_path

print(f"Reading Config File {config_file_path}")
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

print("Defining Variables and Creating Directories")

database_file = config['base']['database_file']

columns_of_interest = config['base']['columns_of_interest']

tag = config['base']['tag']

out_dir_figures = f"outputs/{tag}/figures"
out_dir_stats = f"outputs/{tag}/stats"
out_dir_log = f"outputs/{tag}/log"

# Create the directories
print(create_directory_if_not_exists(out_dir_figures))
print(create_directory_if_not_exists(out_dir_stats))
print(create_directory_if_not_exists(out_dir_log))

print("Initializing Logger")

logger = Logger(config)

logger.log("Loading Data")
# Defining the connection to the database
conn = sqlite3.connect(database_file)

# Loading descriptions into dataframe
description_fetch_query = f"""SELECT *
                    FROM descriptions
                    """
descriptions = pd.read_sql_query(description_fetch_query, conn, index_col = 'name')

column_types = {idx:transform_type(row['data_type']) for idx, row in descriptions.iterrows() if row['location'] == 'loans_data' and idx in columns_of_interest}

conditions_string = f'WHERE {columns_of_interest[0]} IS NOT NULL ' + ' '.join([f'AND {col} IS NOT NULL' for col in columns_of_interest[1:]])
# Loading data into dataframe
data_fetch_query = f"""SELECT {', '.join(columns_of_interest)} 
                       FROM loans_data
                       {conditions_string}
                       ORDER BY RANDOM()"""
loans_data = pd.read_sql_query(data_fetch_query, conn, index_col='id', dtype=column_types)

# Closing connection
conn.close()

logger.log("Creating Proper Columns")
loans_data['mort_acc'] = loans_data['mort_acc'].astype('float').astype('int')
loans_data['pub_rec_bankruptcies'] = loans_data['pub_rec_bankruptcies'].astype('float').astype('int')
loans_data['revol_util'] = loans_data['revol_util'].astype('float')
loans_data['term_months'] = loans_data['term_months'].astype('int')
descriptions = pd.concat([descriptions, pd.DataFrame({
    'name':['issue_d'],
    'full_name': ['Issue Date'],
    'type': ['Column'],
    'location': ['loans_data'],
    'description': ['Date the loan is issued'],
    'data_type': ['TEXT']
}).set_index('name')])

loans_data['issue_d'] = pd.to_datetime(loans_data['issue_d_unix'], unit='s')
loans_data['earliest_cr_line'] = pd.to_datetime(loans_data['earliest_cr_line'])
loans_data['issue_month'] = loans_data['issue_d'].apply(lambda x: x.month)
descriptions = pd.concat([descriptions, pd.DataFrame({
    'name':['issue_month'],
    'full_name': ['Issue Month'],
    'type': ['Column'],
    'location': ['loans_data'],
    'description': ['Month of the year the loan was issued'],
    'data_type': ['INT']
}).set_index('name')])

loans_data['issue_year'] = loans_data['issue_d'].apply(lambda x: x.year)
descriptions = pd.concat([descriptions, pd.DataFrame({
    'name':['issue_year'],
    'full_name': ['Issue Year'],
    'type': ['Column'],
    'location': ['loans_data'],
    'description': ['Year the loan was issued'],
    'data_type': ['INT']
}).set_index('name')])

loans_data = loans_data[(loans_data['loan_status'] == 'Charged Off') | (loans_data['loan_status'] == 'Fully Paid')]

logger.log("Computing Numerical Statistics for Interesting Columns")
# Calculating descriptive statistics for the numerical columns
numerical_stats = loans_data.describe().transpose()

# Adding description of each row
numerical_stats['description'] = [descriptions.loc[name]['description'] for name in numerical_stats.index]
# Displaying the descriptive statistics
numerical_stats.to_csv(f"{out_dir_stats}/numerical_statistics.csv")

fontsize = config['plotting']['fontsize']
figsize_x = config['plotting']['figure_xsize']
figsize_y = config['plotting']['figure_ysize']

fig, ax = plt.subplots(figsize=[10, 10/1.62])

# Define the order of the grades and sub-grades
loan_status_order = sorted(loans_data['loan_status'].unique())[::-1]

logger.log("Plotting Loan_Status_Distribution")
# Plot the counts of each grade with hue set to 'loan_status'
sns.countplot(data=loans_data, x='loan_status', ax=ax, order=loan_status_order)
ax.set_xlabel(descriptions.loc['loan_status']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loan Status Distribution', fontsize=fontsize)
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/01-Loan_Status_Distribution.png')

plt.close("all")

# Filter the DataFrame to include only numerical columns
numerical_data = loans_data.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numerical_data.corr()

# Get the full names for each column
full_names = [descriptions.loc[col]['full_name'] for col in numerical_data.columns]

logger.log("Plotting Correlation_Matrix")
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(len(numerical_data.columns)*1, len(numerical_data.columns)*1))

# Draw the heatmap with the axis object
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, ax=ax)

# Set the tick labels as the full names
ax.set_xticklabels(full_names, rotation=45, ha="right")
ax.set_yticklabels(full_names, rotation=0)
ax.set_title("Correlation Matrix", fontsize=fontsize)

fig.tight_layout()

fig.savefig(f'{out_dir_figures}/02-Correlation_Matrix.png')

plt.close("all")

# Your existing setup
loan_status_values = loans_data['loan_status'].unique()

logger.log("Plotting Distribution_of_Installments_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

best_bw_installment = bestbandwidth(loans_data['installment'])
nBins_installment = int((loans_data['installment'].max() - loans_data['installment'].min())/best_bw_installment)
bins_installment = np.linspace(loans_data['installment'].min(), loans_data['installment'].max(), nBins_installment)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['installment'], label=loan_status, ax=ax, element="step", bins=bins_installment, kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['installment']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.legend()
ax.set_title("Distribution of Installments By Loan Status", fontsize=fontsize)

fig.tight_layout()

fig.savefig(f'{out_dir_figures}/03-Distribution_of_Installments_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Loan_Amount_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

best_bw_loan_amnt = bestbandwidth(loans_data['loan_amnt'])
nBins_loan_amnt = int((loans_data['loan_amnt'].max() - loans_data['loan_amnt'].min())/best_bw_loan_amnt)
bins_loan_amnt = np.linspace(loans_data['loan_amnt'].min(), loans_data['loan_amnt'].max(), nBins_loan_amnt)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['loan_amnt'], label=loan_status, ax=ax, element="step", bins=bins_loan_amnt, kde=False, alpha=0.5)
ax.set_xlabel(descriptions.loc['loan_amnt']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Loan Amount by Loan Status", fontsize=fontsize)
ax.legend()

fig.tight_layout()

fig.savefig(f'{out_dir_figures}/04-Distribution_of_Loan_Amount_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Interest_Rate_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

best_bw_int_rate = bestbandwidth(loans_data['int_rate'])
nBins_int_rate = int((loans_data['int_rate'].max() - loans_data['int_rate'].min())/best_bw_int_rate)
bins_int_rate = np.linspace(loans_data['int_rate'].min(), loans_data['int_rate'].max(), nBins_int_rate)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['int_rate'], label=loan_status, ax=ax, element="step", bins=bins_int_rate, kde=False, alpha=0.5)
ax.set_xlabel(descriptions.loc['int_rate']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.legend()
ax.set_title("Distribution of Interest Rate by Loan Status", fontsize=fontsize)
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/05-Distribution_of_Interest_Rate_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Annual_Income_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

best_bw_annual_inc = bestbandwidth(loans_data['annual_inc'])
nBins_annual_inc = int((loans_data['annual_inc'].max() - loans_data['annual_inc'].min())/best_bw_annual_inc)
bins_annual_inc = np.linspace(loans_data['annual_inc'].min(), loans_data['annual_inc'].max(), nBins_annual_inc)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['annual_inc'], label=loan_status, ax=ax, element="step", bins=bins_annual_inc, kde=False, alpha=0.5)
ax.set_xlabel(descriptions.loc['annual_inc']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.legend()
ax.set_title("Distribution of Annual Income by Loan Status", fontsize=fontsize)
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/06-Distribution_of_Annual_Income_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Violin_of_Installments_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

# Violin plot for 'installment'
sns.violinplot(x='loan_status', y='installment', hue='loan_status', data=loans_data, ax=ax)
ax.set_title('Installment by Loan Status', fontsize=fontsize)
ax.set_xlabel(descriptions.loc['loan_status']['full_name'], fontsize=fontsize)
ax.set_ylabel(descriptions.loc['installment']['full_name'], fontsize=fontsize)
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/07-Violin_of_Installments_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Violin_of_Loan_Amount_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

sns.violinplot(x='loan_status', y='loan_amnt', hue='loan_status', data=loans_data, ax=ax)
ax.set_title('Loan Amount by Loan Status', fontsize=fontsize)
ax.set_xlabel(descriptions.loc['loan_status']['full_name'], fontsize=fontsize)
ax.set_ylabel(descriptions.loc['loan_amnt']['full_name'], fontsize=fontsize)

fig.tight_layout()

fig.savefig(f'{out_dir_figures}/08-Violin_of_Loan_Amount_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Loan_Grade_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

grade_order = sorted(loans_data['grade'].unique())

sns.countplot(data=loans_data, x='grade', hue='loan_status', ax=ax, alpha=0.7, order=grade_order)
ax.set_xlabel(descriptions.loc['grade']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loan Grade Distribution by Loan Status', fontsize=fontsize)
ax.set_title("Distribution of Loan Grade by Loan Status", fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/09-Distribution_of_Loan_Grade_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Loan_Sub_Grade_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

sub_grade_order = sorted(loans_data['sub_grade'].unique())

sns.countplot(data=loans_data, x='sub_grade', hue='loan_status', ax=ax, alpha=0.7, order=sub_grade_order)
ax.set_xlabel(descriptions.loc['sub_grade']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Distribution of Loan Sub-Grade Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/10-Distribution_of_Loan_Sub_Grade_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Home_Ownership_Status_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

home_ownership_order = sorted(loans_data['home_ownership'].unique())

sns.countplot(data=loans_data, x='home_ownership', hue='loan_status', ax=ax, order=home_ownership_order)
ax.set_xlabel(descriptions.loc['home_ownership']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loan Home Ownership Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/11-Distribution_of_Home_Ownership_Status_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Income_Verification_Status_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

verification_status_order = sorted(loans_data['verification_status'].unique())

sns.countplot(data=loans_data, x='verification_status', hue='loan_status', ax=ax, order=verification_status_order)
ax.set_xlabel(descriptions.loc['verification_status']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Income Verification Status Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/12-Distribution_of_Income_Verification_Status_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Term_Length_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

term_months_order = sorted(loans_data['term_months'].unique())

sns.countplot(data=loans_data, x='term_months', hue='loan_status', ax=ax, order=term_months_order)
ax.set_xlabel(descriptions.loc['term_months']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Term Length Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/13-Distribution_of_Term_Length_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Loan_Purpose_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

purpose_order = sorted(loans_data['purpose'].unique())

sns.countplot(data=loans_data, x='purpose', hue='loan_status', ax=ax, order=purpose_order)
ax.set_xlabel(descriptions.loc['purpose']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Purpose Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
ax.tick_params(axis='x', rotation=90)
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/14-Distribution_of_Loan_Purpose_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Issue_Date_By_Loan_Status")
fig, ax = plt.subplots(figsize = [10, 10/1.62])

for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['issue_d'], label=loan_status, ax=ax, element="step", kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['issue_d_unix']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Issue Date by Loan Status", fontsize=fontsize)
ax.legend()
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/15-Distribution_of_Issue_Date_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Earliest_Credit_Line_Date_By_Loan_Status")
fig, ax = plt.subplots(figsize = [10, 10/1.62])

for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['earliest_cr_line'], label=loan_status, ax=ax, element="step", kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['earliest_cr_line']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Earliest Credit Line Date by Loan Status", fontsize=fontsize)
ax.legend()
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/16-Distribution_of_Earliest_Credit_Line_Date_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Debt_To_Income_By_Loan_Status")
fig, ax = plt.subplots(figsize = [10, 10/1.62])

best_bw_dti = bestbandwidth(loans_data['dti'])
nBins_dti = int((loans_data['dti'].max() - loans_data['dti'].min())/best_bw_dti)
bins_dti = np.linspace(loans_data['dti'].min(), loans_data['dti'].max(), nBins_dti)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['dti'], label=loan_status, ax=ax, element="step", bins=bins_dti, kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['dti']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Debt To Income Ratio by Loan Status", fontsize=fontsize)
ax.legend()
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/17-Distribution_of_Debt_To_Income_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Number_Of_Open_Accounts_By_Loan_Status")
fig, ax = plt.subplots(figsize = [10, 10/1.62])

best_bw_open_acc = bestbandwidth(loans_data['open_acc'])
nBins_open_acc = int((loans_data['open_acc'].max() - loans_data['open_acc'].min())/best_bw_open_acc)
bins_open_acc = np.linspace(loans_data['open_acc'].min(), loans_data['open_acc'].max(), nBins_open_acc)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['open_acc'], label=loan_status, ax=ax, element="step", bins=bins_open_acc, kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['open_acc']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Number of Open Accounts by Loan Status", fontsize=fontsize)
ax.legend()
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/18-Distribution_of_Number_Of_Open_Accounts_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Revol_Util_By_Loan_Status")
fig, ax = plt.subplots(figsize = [10, 10/1.62])

best_bw_revol_util = bestbandwidth(loans_data['revol_util'])
nBins_revol_util = int((loans_data['revol_util'].max() - loans_data['revol_util'].min())/best_bw_revol_util)
bins_revol_util = np.linspace(loans_data['revol_util'].min(), loans_data['revol_util'].max(), nBins_revol_util)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['revol_util'], label=loan_status, ax=ax, element="step",bins=bins_revol_util, kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['revol_util']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Revolving Line Utilization Rate by Loan Status", fontsize=fontsize)
ax.legend()
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/19-Distribution_of_Revol_Util_By_Loan_Status.png')

plt.close("all")

logger.log("Plotting Distribution_of_Revol_Bal_By_Loan_Status")
fig, ax = plt.subplots(figsize = [10, 10/1.62])

best_bw_revol_bal = bestbandwidth(loans_data['revol_bal'])
nBins_revol_bal = int((loans_data['revol_bal'].max() - loans_data['revol_bal'].min())/best_bw_revol_bal)
bins_revol_bal = np.linspace(loans_data['revol_bal'].min(), loans_data['revol_bal'].max(), nBins_revol_bal)
for loan_status in loan_status_values:
    mask = loans_data['loan_status'] == loan_status
    sns.histplot(loans_data[mask]['revol_bal'], label=loan_status, ax=ax, element="step", bins=bins_revol_bal, kde=False, alpha=0.5, edgecolor='b')
ax.set_xlabel(descriptions.loc['revol_bal']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title("Distribution of Revolving Balance by Loan Status", fontsize=fontsize)
ax.legend()
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/20-Distribution_of_Revol_Bal_By_Loan_Status.png')

plt.close("all")

pub_rec_order = sorted(loans_data['pub_rec'].unique())

logger.log("Plotting Distribution_of_Number_of_Public_Records_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

sns.countplot(data=loans_data, x='pub_rec', hue='loan_status', ax=ax, order=pub_rec_order)
ax.set_xlabel(descriptions.loc['pub_rec']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loaner Number of Public Records Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/21-Distribution_of_Number_of_Public_Records_By_Loan_Status.png')

plt.close("all")

initial_list_status_order = sorted(loans_data['initial_list_status'].unique())

logger.log("Plotting Distribution_of_Initial_Listing_Status_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

sns.countplot(data=loans_data, x='initial_list_status', hue='loan_status', ax=ax, order=initial_list_status_order)
ax.set_xlabel(descriptions.loc['initial_list_status']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loaner Initial Listing Status Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/22-Distribution_of_Initial_Listing_Status_By_Loan_Status.png')

plt.close("all")

application_type_order = sorted(loans_data['application_type'].unique())

logger.log("Plotting Distribution_of_Application_Type_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

sns.countplot(data=loans_data, x='application_type', hue='loan_status', ax=ax, order=application_type_order)
ax.set_xlabel(descriptions.loc['application_type']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loan Application Type Distribution by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/23-Distribution_of_Application_Type_By_Loan_Status.png')

plt.close("all")

pub_rec_bankruptcies_order = sorted(loans_data['pub_rec_bankruptcies'].unique())

logger.log("Plotting Distribution_of_Number_of_Public_Record_Bankruptcies_By_Loan_Status")
fig, ax = plt.subplots(figsize=[10, 10/1.62])

sns.countplot(data=loans_data, x='pub_rec_bankruptcies', hue='loan_status', ax=ax, order=pub_rec_bankruptcies_order)
ax.set_xlabel(descriptions.loc['pub_rec_bankruptcies']['full_name'], fontsize=fontsize)
ax.set_ylabel('Counts', fontsize=fontsize)
ax.set_title('Loaner Number of Public Record Bankruptcies by Loan Status', fontsize=fontsize)
ax.legend(title=descriptions.loc['loan_status']['full_name'])
fig.tight_layout()

fig.savefig(f'{out_dir_figures}/25-Distribution_of_Number_of_Public_Record_Bankruptcies_By_Loan_Status.png')

plt.close("all")

loans_data['loan_status_num'] = loans_data['loan_status'].apply(lambda x: 0 if x == 'Charged Off' else 1)

num_cols = ['mort_acc', 'annual_inc', 'loan_status_num', 'total_acc', 'revol_bal', 'pub_rec_bankruptcies', 'pub_rec', 'open_acc', 'installment', 'loan_amnt', 'dti', 'revol_util', 'int_rate']

correlation = loans_data[num_cols].corr()['loan_status_num'].drop('loan_status_num').sort_values().iloc[::-1]

full_names_map = {col: descriptions.loc[col]['full_name'] for col in num_cols if col in descriptions.index}

full_names_map['loan_status_num'] = 'Numerical Loan Status'  # Replace with the actual full name

correlation.index = [full_names_map.get(col, col) for col in correlation.index]

logger.log("Plotting Correlation_Between_Loan_Status_And_Numerical_Features")
fig, ax = plt.subplots(figsize=[10, 10/1.62])
sns.barplot(x=correlation.values, y=correlation.index, edgecolor="black", ax=ax)

ax.set_title("Correlation between Loan status and Numeric Features", fontsize=fontsize)
ax.set_xlabel('Correlation')
ax.set_ylabel('Numerical Features')

fig.tight_layout()

fig.savefig(f'{out_dir_figures}/26-Correlation_Between_Loan_Status_And_Numerical_Features.png')

plt.close("all")

logger.log("Done!")
