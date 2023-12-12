# Automated Exploratory Data Analysis Report
Run Tag: test_tag
Author: Andre Guimaraes
## numerical statistics
|                      |       count | mean                          | min                 | 25%                 | 50%                 | 75%                 | max                 |          std | description                                                                                                                     |
|:---------------------|------------:|:------------------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|-------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| loan_amnt            | 1.29686e+06 | 14522.782164009866            | 1000.0              | 8000.0              | 12000.0             | 20000.0             | 40000.0             |  8734.65     | The listed amount of the loan applied for by the borrower.                                                                      |
| term_months          | 1.29686e+06 | 41.775773964981596            | 36.0                | 36.0                | 36.0                | 36.0                | 60.0                |    10.2596   | The length of the loan term in months.                                                                                          |
| int_rate             | 1.29686e+06 | 13.279633885204358            | 5.31                | 9.75                | 12.79               | 16.02               | 30.99               |     4.79411  | Interest rate of the loan.                                                                                                      |
| installment          | 1.29686e+06 | 441.6820788426826             | 4.93                | 251.36              | 377.41              | 585.63              | 1719.83             |   262.204    | The monthly payment owed by the borrower if the loan originates.                                                                |
| annual_inc           | 1.29686e+06 | 76521.22443962768             | 16.0                | 46000.0             | 65000.0             | 90480.0             | 10999200.0          | 70151        | The self-reported annual income provided by the borrower during registration.                                                   |
| dti                  | 1.29686e+06 | 18.458561356999706            | -1.0                | 11.95               | 17.78               | 24.29               | 999.0               |    11.2533   | Debt-to-income ratio calculated using the borrower’s total monthly debt payments divided by their self-reported monthly income. |
| earliest_cr_line     | 1.29686e+06 | 1999-03-30 00:24:56.405859968 | 1934-04-01 00:00:00 | 1995-05-01 00:00:00 | 2000-08-01 00:00:00 | 2004-07-01 00:00:00 | 2015-10-01 00:00:00 |   nan        | The month and year the borrower’s earliest reported credit line was opened.                                                     |
| open_acc             | 1.29686e+06 | 11.680264114658394            | 1.0                 | 8.0                 | 11.0                | 14.0                | 90.0                |     5.49133  | The number of open credit lines in the borrower’s credit file.                                                                  |
| pub_rec              | 1.29686e+06 | 0.22136759452246618           | 0.0                 | 0.0                 | 0.0                 | 0.0                 | 86.0                |     0.61052  | Number of derogatory public records, such as bankruptcy filings, tax liens, or judgments.                                       |
| revol_bal            | 1.29686e+06 | 16349.187597591415            | 0.0                 | 6014.0              | 11199.0             | 19842.0             | 2904836.0           | 22517.3      | The total amount of credit revolving balances.                                                                                  |
| revol_util           | 1.29686e+06 | 51.87276932531705             | 0.0                 | 33.6                | 52.2                | 70.6                | 892.3               |    24.3784   | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.      |
| total_acc            | 1.29686e+06 | 25.09046304885412             | 2.0                 | 16.0                | 23.0                | 32.0                | 176.0               |    12.0112   | The total number of credit lines currently in the borrower’s credit file.                                                       |
| mort_acc             | 1.29686e+06 | 1.6706285407611148            | 0.0                 | 0.0                 | 1.0                 | 3.0                 | 51.0                |     2.00042  | Number of mortgage accounts.                                                                                                    |
| pub_rec_bankruptcies | 1.29686e+06 | 0.1378937295515865            | 0.0                 | 0.0                 | 0.0                 | 0.0                 | 12.0                |     0.382491 | Number of public record bankruptcies.                                                                                           |
| issue_d              | 1.29686e+06 | 2015-08-04 01:53:30.061834496 | 2012-03-01 06:00:00 | 2014-08-01 05:00:00 | 2015-09-01 05:00:00 | 2016-07-01 05:00:00 | 2018-12-01 06:00:00 |   nan        | Date the loan is issued                                                                                                         |
| issue_month          | 1.29686e+06 | 6.514445264372974             | 1.0                 | 3.0                 | 7.0                 | 10.0                | 12.0                |     3.4417   | Month of the year the loan was issued                                                                                           |
| issue_year           | 1.29686e+06 | 2015.1304002510678            | 2012.0              | 2014.0              | 2015.0              | 2016.0              | 2018.0              |     1.42295  | Year the loan was issued                                                                                                        |
## Violin of Loan Amount By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/08-Violin_of_Loan_Amount_By_Loan_Status.png)
## Distribution of Earliest Credit Line Date By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/16-Distribution_of_Earliest_Credit_Line_Date_By_Loan_Status.png)
## Distribution of Loan Grade By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/09-Distribution_of_Loan_Grade_By_Loan_Status.png)
## Violin of Installments By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/07-Violin_of_Installments_By_Loan_Status.png)
## Distribution of Interest Rate By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/05-Distribution_of_Interest_Rate_By_Loan_Status.png)
## Distribution of Revol Util By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/19-Distribution_of_Revol_Util_By_Loan_Status.png)
## Distribution of Loan Amount By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/04-Distribution_of_Loan_Amount_By_Loan_Status.png)
## Distribution of Issue Date By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/15-Distribution_of_Issue_Date_By_Loan_Status.png)
## Distribution of Loan Purpose By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/14-Distribution_of_Loan_Purpose_By_Loan_Status.png)
## Distribution of Revol Bal By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/20-Distribution_of_Revol_Bal_By_Loan_Status.png)
## Distribution of Annual Income By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/06-Distribution_of_Annual_Income_By_Loan_Status.png)
## Distribution of Term Length By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/13-Distribution_of_Term_Length_By_Loan_Status.png)
## Loan Status Distribution
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/01-Loan_Status_Distribution.png)
## numerical statistics
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/00-numerical_statistics.png)
## Distribution of Loan Sub Grade By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/10-Distribution_of_Loan_Sub_Grade_By_Loan_Status.png)
## Distribution of Debt To Income By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/17-Distribution_of_Debt_To_Income_By_Loan_Status.png)
## Distribution of Income Verification Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/12-Distribution_of_Income_Verification_Status_By_Loan_Status.png)
## Distribution of Number Of Open Accounts By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/18-Distribution_of_Number_Of_Open_Accounts_By_Loan_Status.png)
## Distribution of Application Type By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/23-Distribution_of_Application_Type_By_Loan_Status.png)
## Distribution of Number of Public Records By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/21-Distribution_of_Number_of_Public_Records_By_Loan_Status.png)
## Distribution of Initial Listing Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/22-Distribution_of_Initial_Listing_Status_By_Loan_Status.png)
## Distribution of Number of Public Record Bankruptcies By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/25-Distribution_of_Number_of_Public_Record_Bankruptcies_By_Loan_Status.png)
## Correlation Matrix
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/02-Correlation_Matrix.png)
## Distribution of Installments By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/03-Distribution_of_Installments_By_Loan_Status.png)
## Correlation Between Loan Status And Numerical Features
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/26-Correlation_Between_Loan_Status_And_Numerical_Features.png)
## Distribution of Home Ownership Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/test_tag/figures/11-Distribution_of_Home_Ownership_Status_By_Loan_Status.png)
