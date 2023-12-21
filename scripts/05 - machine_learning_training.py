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
import ast
from collections import OrderedDict
import pickle

# ML imports
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC as SupportVectorClassifier

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
    
def string_to_dict(string_dict):
    tuple_list_str = string_dict.replace("OrderedDict", "")
    # Safely evaluate the string to a list of tuples using literal_eval
    tuple_list = ast.literal_eval(tuple_list_str)
    # Convert the list of tuples to an OrderedDict
    actual_ordered_dict = OrderedDict(tuple_list)
    # Convert OrderedDict to a regular dict
    params = dict(actual_ordered_dict)
    return params

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
n_jobs = config['machine_learning']['n_jobs']

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

cummulative_results = pd.read_csv(os.path.join(out_dir_stats, 'Cummulative_Results_BayesSearchCV.csv'), index_col=0)

model = cummulative_results.sort_values('mean_test_score').iloc[-1]['ML_model']
params = string_to_dict(cummulative_results.sort_values('mean_test_score').iloc[-1]['params'])
params['verbose'] = 1
params['random_state'] = random_state
params['n_jobs'] = n_jobs

logger.log("Loading best parameters and defining")

if model == 'RandomForestClassifier':
    clf = RandomForestClassifier(**params)
elif model == 'LogisticRegression':
    clf = LogisticRegression(**params)
elif model == 'SupportVectorClassifier':
    clf = SupportVectorClassifier(**params)

logger.log("Fitting the model to the training data")
clf.fit(X_train, y_train)

logger.log("Saving Model")

model_file_path = os.path.join(out_dir_models, 'best_model.pkl')

with open(model_file_path, 'wb') as f:
    pickle.dump(clf, f)

logger.log("Making predictions with the fit model")
y_pred = clf.predict(X_test)

logger.log("Computing cross-validations")

train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, n_jobs=1, 
    train_sizes=np.linspace(.1, 1.0, 10),
    verbose=False)

logger.log("Computing mean scores")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

logger.log(f"Recall score: {recall_score(y_test, y_pred)}")

logger.log("Computing Importance of Features")

if model == 'RandomForestClassifier':
    feature_importances = clf.feature_importances_

    features = np.array(ML_columns)  # Assuming ML_columns are your feature names
    importances = feature_importances

    # Sorting features by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = features[sorted_indices]
    sorted_importances = importances[sorted_indices]
elif model == 'LogisticRegression':
    # Logistic Regression coefficients
    coefficients = clf.coef_[0]  # Assuming clf is your Logistic Regression model

    features = np.array(ML_columns)  # Assuming ML_columns are your feature names
    importances = np.abs(coefficients)

    # Sorting features by the absolute value of coefficients
    sorted_indices = np.argsort(np.abs(importances))[::-1]
    sorted_features = features[sorted_indices]
    sorted_importances = importances[sorted_indices]
elif model == 'SupportVectorClassifier':
    if clf.kernel == 'linear':
        # SVC coefficients for linear kernel
        coefficients = clf.coef_[0]  # Assuming clf is your SVC model

        features = np.array(ML_columns)  # Assuming ML_columns are your feature names
        importances = np.abs(coefficients)

        # Sorting features by the absolute value of coefficients
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = features[sorted_indices]
        sorted_importances = importances[sorted_indices]
    else:
        # For non-linear kernels, feature importances are not directly available
        sorted_features = None
        sorted_importances = None

logger.log("Computing Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizing the confusion matrix

logger.log("Plotting Confusion Matrix")

fig, ax = plt.subplots(figsize = [figsize_x, figsize_x])

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix', fontsize=fontsize)
ax.set_ylabel('Actual Labels', fontsize=fontsize)
ax.set_xlabel('Predicted Labels', fontsize=fontsize)

fig.tight_layout()

fig.savefig(os.path.join(out_dir_figures, 'Confusion_Matrix.png'), bbox_inches='tight')
plt.close('all')

logger.log("Plotting Learning Curve")

fig, ax = plt.subplots(figsize = [figsize_x, figsize_y])

ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
ax.set_title('Learning Curve', fontsize=fontsize)
ax.set_xlabel('Training Examples', fontsize=fontsize)
ax.set_ylabel('Score', fontsize=fontsize)
ax.legend()

fig.tight_layout()

fig.savefig(os.path.join(out_dir_figures, 'Learning_Curve.png'), bbox_inches='tight')

plt.close('all')

logger.log("Plotting Feature Importances")

fig, ax = plt.subplots(figsize = [figsize_x, figsize_y])

sns.barplot(x=sorted_importances, y=sorted_features, edgecolor="black", ax=ax)
ax.set_title("Feature Importances", fontsize=fontsize)
ax.set_xlabel('Importance Value', fontsize=fontsize)
ax.set_ylabel('Features', fontsize=fontsize)
ax.tick_params(axis='y')

fig.tight_layout()

fig.savefig(os.path.join(out_dir_figures, 'Feature_Importances.png'), bbox_inches='tight')

logger.log("Done!")