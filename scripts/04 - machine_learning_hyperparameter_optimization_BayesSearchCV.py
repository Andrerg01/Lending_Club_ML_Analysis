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
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
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
algorithms = config['machine_learning']['algorithms']

optimization_sample_size = config['machine_learning']['optimization_sample_size']

out_dir_figures = f"outputs/{tag}/figures"
out_dir_log = f"outputs/{tag}/log"
out_dir_stats = f"outputs/{tag}/stats"


sqlite_file = f"outputs/{tag}/data/{sqlite_file}"

columns_of_interest = config['base']['columns_of_interest']

print("Initializing Logger")

# Initialize a Logger instance using the configuration
logger = Logger(config)

logger.log("----------------------Logger Initialized for machine_learing_optimization_BayesSearchCV.py----------------------")

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
   
X_opt, y_opt = sample_data.drop('loan_status', axis='columns').values, sample_data['loan_status'].values

X, y = loans_data.drop('loan_status', axis='columns').values, loans_data['loan_status'].values

ML_columns = loans_data.drop('loan_status', axis='columns').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

cummulative_results = None

logger.log("Staring ML Hyperparameter Optimization")

for algorithm in algorithms:
    logger.log(f"Optimizing {algorithm} Hyperparameters")

    if algorithm == "LogisticRegression":
        clf = LogisticRegression(class_weight='balanced', max_iter = 1000000)
        param_grid = {
            'C': Real(1e-10, 1e10, 'log-uniform'),
            'solver': ['lbfgs'],
            'penalty': ['l2']
            }
        plot_param = 'C'

    elif algorithm == "RandomForestClassifier":
        clf = RandomForestClassifier(class_weight='balanced')

        param_grid = {
        'n_estimators': Integer(10, 1000, 'log-uniform'),
        'criterion': Categorical(['gini', 'entropy']),
        'max_features': Categorical(['sqrt', 'log2']),
        'max_depth': Integer(1, 100),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 10),
        'bootstrap': Categorical([True, False]),
        'class_weight': ['balanced']
        }
        plot_param = 'n_estimators'
    
    elif algorithm == "SupportVectorClassifier":
        clf = SupportVectorClassifier(class_weight='balanced', probability=True)  # probability=True if you need probability estimates

        param_grid = {
            'C': Real(1e-6, 1e+6, 'log-uniform'),
            'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': Integer(1, 5),  # only used with 'poly' kernel
            'gamma': Categorical(['scale', 'auto']),
            'class_weight': ['balanced'],
            'probability': [True]
        }
        plot_param = 'C'

    grid_search = BayesSearchCV(estimator=clf,
                                    search_spaces=param_grid,
                                    cv=5,
                                    verbose=False,
                                    scoring=make_scorer(recall_score),
                                    n_iter=bayes_search_iterations,
                                    n_jobs=-1,
                                    random_state=random_state)
    np.int = int
    grid_search.fit(X_opt, y_opt)

    results = pd.DataFrame(grid_search.cv_results_)
    results['ML_model'] = [algorithm]*len(results)

    if cummulative_results is None:
        cummulative_results = results.copy()
    else:
        cummulative_results = pd.concat([cummulative_results, results.copy()])

    logger.log(f"Plotting Results for {algorithm} Hyperparameter Optimization")

    ml_mask = cummulative_results['ML_model'] == algorithm
    ml_results = cummulative_results[ml_mask]

    fig, ax = plt.subplots(figsize=[figsize_x, figsize_y])

    x_plot = ml_results['params'].apply(lambda x: x[plot_param])
    y_plot = ml_results[f'mean_test_score']
    y_std = ml_results[f'std_test_score']

    ax.errorbar(x_plot, y_plot, yerr=y_std, fmt='o')
    ax.scatter(x_plot, y_plot)
    ax.set_xscale('log')
    ax.set_xlabel(plot_param, fontsize=fontsize)
    ax.set_ylabel(f'Mean Score', fontsize=fontsize)
    ax.set_title(f'{algorithm} Hyperparameter Optimization', fontsize=fontsize)
    ax.legend()

    fig.savefig(os.path.join(out_dir_figures, f"{algorithm}_BayesSearchCV.png"))
    plt.close('all')

logger.log("Saving Preliminary Grid Search Results")

cummulative_results.to_csv(os.path.join(out_dir_stats, "Cummulative_Results_BayesSearchCV.csv"))

logger.log("Done!")