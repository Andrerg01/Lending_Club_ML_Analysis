# All Lending Club Data Analysis

This project brings together scrips for decoding and analysing data from the All Lending Club loand dataset found [here](https://www.kaggle.com/datasets/wordsforthewise/lending-club/)

## Step 01 - Csv to Sqlite

The first step of the process is to re-interpret the data from a csv file to a more efficient sqlite file. This process also cleans the data, redefines redundant/inefficient data types (such as integers being saved as floats or strings), and adds a descriptor for all columns in the dataset as well as a description of the tables involved.

The finals tables are:
 - loans_data: Containing the relevat data for all the loans available
 - metadata: Some metadata provided
 - descriptions: A table of descriptions of tables and columns
 
This is accomplished by the script `01 - csv_to_sqlite.py`, located in the `scripts` directory.
```
python "scripts/01 - csv_to_sqlite.py" --input_csv "data/accepted_2007_to_2018Q4.csv" --output_sqlite "data/All_Lending_Club_Loan_2007_2018.sqlite" 
```

