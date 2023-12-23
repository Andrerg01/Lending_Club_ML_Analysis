# All Lending Club Data Analysis

This comprehensive project compiles scripts for decoding and analyzing data from the Lending Club loan dataset, available [here](https://www.kaggle.com/datasets/wordsforthewise/lending-club/).

## Introduction

This project undertakes a thorough Exploratory Data Analysis (EDA) of the Lending Club dataset and trains a model to predict the likelihood of loan defaults. With the data available at the time of writing, the model achieves an accuracy of approximately 80%. By identifying an optimal threshold to maximize profitability, it forecasts an increase in profitability of about 130% ($706 million across all historical loans).

The workflow is segmented into six primary steps:
 - **01 - CSV to SQLite**: Transforms the data from CSV to a more efficient SQLite file format.
 - **02 - Exploratory Data Analysis**: Generates relevant EDA plots to uncover patterns in the data and key insights. It also produces an automated report.
 - **03 - Data Preprocessing**: Prepares the data for ML optimization and training.
 - **04 - Machine Learning Hyperparameter Optimization**: Conducts a Bayesian Search to fine-tune model hyperparameters, aiming to maximize the recall score.
 - **05 - Machine Learning Training**: Trains the optimal machine learning model from step 04 on the entire dataset and generates plots related to training and predictability.
 - **06 - Machine Learning Results and Final Analysis**: Performs the final analysis of the model, calculating key performance metrics.

All configurations are specified in the `config/config.yml` file. The scripts can be executed using the following commands:
```bash
python "scripts/01 - csv_to_sqlite.py" --config "config/config.yml"  && \
python "scripts/02 - exploratory_data_analysis.py" --config "config/config.yml" && \
python "scripts/03 - data_preprocessing.py" --config "config/config.yml" && \
python "scripts/04 - machine_learning_hyperparameter_optimization_BayesSearchCV.py" --config "config/config.yml" && \
python "scripts/05 - machine_learning_training.py" --config "config/config.yml" && \
python "scripts/06 - machine_learning_results_and_final_analysis.py" --config "config/config.yml" 
```

## Project Showcase

### Machine Learning Performance

Below are three pivotal plots demonstrating the model's predictive capability (confusion matrix), the significance of each feature in the prediction (Feature Importances), and a Learning Curve to verify the model's fitting quality.

<img src="https://github.com/Andrerg01/Lending_Club_ML_Analysis/assets/29161499/287e5d9a-8fbc-4fdc-aa68-0356cc3c13f5" height="225" width="300"/>
<img src="https://github.com/Andrerg01/Lending_Club_ML_Analysis/assets/29161499/396ec2a0-7527-4a00-88fd-2c700235123a" height="225"/>
<img src="https://github.com/Andrerg01/Lending_Club_ML_Analysis/assets/29161499/59406ee8-fe8c-4801-a939-e138bbed2814" height="225"/>

### Key Insights

The model not only classifies loans but also calculates the probability of default (ranging from 0 to 1, or 0% to 100%). Testing various thresholds reveals the proportion of defaulted loans beneath that threshold.

<img src="https://github.com/Andrerg01/Lending_Club_ML_Analysis/assets/29161499/1c05c992-724d-4bde-aca9-dbd15e41d584" height="450"/>

This indicates that lowering the threshold increasingly filters out defaulted loans more than fully paid ones.

A crucial question arises: At what probability threshold should loan applications be rejected to maximize profits? The following figures demonstrate the total and relative profitability at different thresholds, with an optimal point identified.

<img src="https://github.com/Andrerg01/Lending_Club_ML_Analysis/assets/29161499/eae57f00-64cb-486b-9fc1-998c024319d9" height="300"/>
<img src="https://github.com/Andrerg01/Lending_Club_ML_Analysis/assets/29161499/9727b3ef-0e48-449a-9536-567f936903ef" height="300"/>

The optimal threshold of 56% yields a profit margin approximately 130% higher than historical profits by selectively approving loans.

### Exploratory Data Analysis

You can find the [Tableau Dashboard here](https://scholar.google.com/citations?user=4d66dpcAAAAJ&hl=en)
