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
import pickle as pkl

# ML imports
from sklearn.model_selection import train_test_split

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
    
def calculate_profit(df, threshold):
    # Predicted defaults based on threshold
    return df[df['Default Probability'] <= threshold]['Profit_or_Loss'].sum()

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
out_dir_models = f"outputs/{tag}/models"


sqlite_file = f"outputs/{tag}/data/{sqlite_file}"

columns_of_interest = config['base']['columns_of_interest']

print("Initializing Logger")

# Initialize a Logger instance using the configuration
logger = Logger(config)

logger.log("----------------------Logger Initialized for machine_learning_results_and_final_analysis.py----------------------")

logger.log("Loading Data")
# Defining the connection to the database
conn = sqlite3.connect(sqlite_file)

# Loading data into dataframe
data_fetch_query = f"""SELECT * 
                       FROM loans_data_ML
                       ORDER BY RANDOM()"""

loans_data = pd.read_sql_query(data_fetch_query, conn, index_col='id')

# Loading data into dataframe
data_fetch_query = f"""SELECT id, total_pymnt
                       FROM loans_data"""

loans_data_paymnts = pd.read_sql_query(data_fetch_query, conn)

loans_data_paymnts = loans_data_paymnts[loans_data_paymnts['id'].apply(lambda x: x in loans_data.index)]

combined_data = pd.merge(loans_data, loans_data_paymnts, on='id', how='inner')
combined_data['Profit_or_Loss'] = combined_data['total_pymnt'] - combined_data['loan_amnt']

# Closing connection
conn.close()

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    # Since we're setting this at the sys level, it should not be overridden
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

X, y = combined_data.drop(['loan_status', 'Profit_or_Loss', 'total_pymnt'], axis='columns').values, combined_data['loan_status'].values

ML_columns = combined_data.drop('loan_status', axis='columns').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

logger.log("Loading Model")

with open(os.path.join(out_dir_models, 'best_model.pkl'), 'rb') as f:
    model = pkl.load(f)

logger.log("Computing Probabilities")

probabilities = model.predict_proba(combined_data.drop(['id', 'total_pymnt', 'loan_status', 'Profit_or_Loss'], axis='columns'))
combined_data['Default Probability'] = [prob[1] for prob in probabilities]

logger.log("Computing Proportion and Quantity Curves")

threshs = np.linspace(0, 1, 100)
prop = np.array([0. for _ in range(len(threshs))])
N = np.array([0 for _ in range(len(threshs))])
N_defaulted = np.array([0 for _ in range(len(threshs))])
N_paid = np.array([0 for _ in range(len(threshs))])
for i, thresh in enumerate(threshs):
    thresh_mask = (combined_data['Default Probability'] <= thresh)
    combined_data_thresh = combined_data[thresh_mask]
    if len(combined_data_thresh) == 0:
        prop[i] = np.nan
    else:
        prop[i] = len(combined_data_thresh[combined_data_thresh['loan_status'] == 1])/len(combined_data_thresh)
    N[i] = len(combined_data_thresh)
    N_defaulted[i] = len(combined_data_thresh[combined_data_thresh['loan_status'] == 1])
    N_paid[i] = len(combined_data_thresh[combined_data_thresh['loan_status'] == 0])

fig, ax = plt.subplots(figsize = [figsize_x, figsize_y])
ax.plot(threshs, prop, label='Proportion of Defaulted Loans')
ax.set_xlabel('Threshold on Default Probability', fontsize=fontsize)
ax.set_ylabel('Proportion of Defaulted Loans Below Threshold', fontsize=fontsize)
ax.legend(loc='upper left')

ax2 = ax.twinx()
# ax2.bar(threshs, N, width=np.diff(threshs)[0], color=None, edgecolor='k', alpha=0.1)
ax2.bar(threshs, N_defaulted, width=np.diff(threshs)[0], color='Red', edgecolor='k', alpha=0.2, label='Defaulted')
ax2.bar(threshs, N_paid, width=np.diff(threshs)[0], color='Blue', edgecolor='k', alpha=0.2, label='Paid Off', bottom=N_defaulted)
ax2.set_ylabel("Number of Loans Below Threshold", fontsize=fontsize)
ax2.legend(loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(out_dir_figures, 'Proportion_Quantity_Curves.png'))
plt.close('all')

logger.log("Computing Profit Curves")

actual_profit = combined_data['Profit_or_Loss'].sum()
# Evaluate profits at various thresholds
thresholds = np.linspace(0, 1, 100)
profits = np.array([calculate_profit(combined_data, thresh) for thresh in thresholds])
Ns = np.array([len(combined_data[combined_data['Default Probability'] <= thresh]) for thresh in thresholds])
Ns_defaulted = np.array([len(combined_data[(combined_data['Default Probability'] <= thresh) & (combined_data['loan_status'] == 1)]) for thresh in thresholds])
Ns_paid = np.array([len(combined_data[(combined_data['Default Probability'] <= thresh) & (combined_data['loan_status'] == 0)]) for thresh in thresholds])

# Find the optimal threshold
max_profit = profits.max()
optimal_threshold = thresholds[profits.argmax()]
print(f"Maximum profit of {max_profit} is achieved at a threshold of {optimal_threshold:.2f}")

# Plot the profit curve using fig, ax
fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
ax.plot(thresholds, profits, label='Profit')
ax.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
ax.axhline(actual_profit, color='blue', linestyle='--', label='Unaltered Profit')
ax.set_xlabel('Default Probability Threshold', fontsize=fontsize)
ax.set_ylabel('Total Profit Considering Only Loans Below Threshold', fontsize=fontsize)
ax.set_title('Profitability Considering Only Loans Below Threshold', fontsize=fontsize)
ax.legend(loc='upper left')
ax.grid(True)

ax2 = ax.twinx()
# ax2.bar(thresholds, Ns, width=np.diff(thresholds)[0], color=None, edgecolor='k', alpha=0.1)
ax2.bar(thresholds, Ns_defaulted, width=np.diff(thresholds)[0], color='Red', edgecolor='k', alpha=0.2, label='Defaulted')
ax2.bar(thresholds, Ns_paid, width=np.diff(thresholds)[0], color='Blue', edgecolor='k', alpha=0.2, label='Paid Off', bottom=Ns_defaulted)
ax2.set_ylabel('Number of Loans With Default Probability Below Threshold', fontsize=fontsize)
ax2.legend(loc='upper right')

fig.tight_layout()

fig.savefig(os.path.join(out_dir_figures, 'Profit_Curve.png'))
plt.close('all')

relative_profit = profits/actual_profit

# Find the optimal threshold
max_relative_profit = relative_profit.max()
optimal_threshold = thresholds[relative_profit.argmax()]
logger.log(f"Maximum Relative Profit of {max_relative_profit} is achieved at a threshold of {optimal_threshold:.2f}")

# Plot the profit curve using fig, ax
fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
ax.plot(thresholds, relative_profit, label='Profit')
ax.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
ax.axhline(1, color='blue', linestyle='--', label='Unaltered Profit')
ax.set_xlabel('Default Probability Threshold')
ax.set_ylabel('Total Profit Considering Only Loans Below Threshold')
ax.set_title('Profitability Considering Only Loans Below Threshold')
ax.legend(loc='upper left')
ax.grid(True)

ax2 = ax.twinx()
# ax2.bar(thresholds, Ns, width=np.diff(thresholds)[0], color=None, edgecolor='k', alpha=0.1)
ax2.bar(thresholds, Ns_defaulted, width=np.diff(thresholds)[0], color='Red', edgecolor='k', alpha=0.2, label='Defaulted')
ax2.bar(thresholds, Ns_paid, width=np.diff(thresholds)[0], color='Blue', edgecolor='k', alpha=0.2, label='Paid Off', bottom=Ns_defaulted)
ax2.set_ylabel('Number of Loans With Probability Default Below Threshold')
ax2.legend(loc='upper right')

fig.tight_layout()

fig.savefig(os.path.join(out_dir_figures, 'Relative_Profit_Curve.png'))
plt.close('all')

logger.log("Done!")
