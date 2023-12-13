# Automated Exploratory Data Analysis Report
Run Tag: prototype
Author: Andre Guimaraes
2023-12-13 15:15:33
## numerical statistics
|                      |       count | mean                          | min                 | 25%                 | 50%                 | 75%                 | max                 |             std | description                                                                                                                     |
|:---------------------|------------:|:------------------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|----------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| loan_amnt            | 1.34531e+06 | 14419.972013885275            | 500.0               | 8000.0              | 12000.0             | 20000.0             | 40000.0             |  8717.05        | The listed amount of the loan applied for by the borrower.                                                                      |
| term_months          | 1.34531e+06 | 41.790195568307674            | 36.0                | 36.0                | 36.0                | 36.0                | 60.0                |    10.2683      | The length of the loan term in months.                                                                                          |
| int_rate             | 1.34531e+06 | 13.239618925658709            | 5.309999942779541   | 9.75                | 12.739999771118164  | 15.989999771118164  | 30.989999771118164  |     4.76872     | Interest rate of the loan.                                                                                                      |
| installment          | 1.34531e+06 | 438.0755328540502             | 4.929999828338623   | 248.47999572753906  | 375.42999267578125  | 580.72998046875     | 1719.8299560546875  |   261.513       | The monthly payment owed by the borrower if the loan originates.                                                                |
| annual_inc           | 1.34531e+06 | 76247.63641372623             | 0.0                 | 45780.0             | 65000.0             | 90000.0             | 10999200.0          | 69925.1         | The self-reported annual income provided by the borrower during registration.                                                   |
| issue_d_unix         | 1.34531e+06 | 1433612900.8350492            | 1180656000.0        | 1404172800.0        | 1438387200.0        | 1467331200.0        | 1543622400.0        |     5.10782e+07 | The date when the loan was funded, converted to Unix time.                                                                      |
| dti                  | 1.34531e+06 | 18.277862284080516            | -1.0                | 11.789999961853027  | 17.610000610351562  | 24.049999237060547  | 999.0               |    11.1626      | Debt-to-income ratio calculated using the borrower’s total monthly debt payments divided by their self-reported monthly income. |
| open_acc             | 1.34531e+06 | 11.593520452535103            | 0.0                 | 8.0                 | 11.0                | 14.0                | 90.0                |     5.47379     | The number of open credit lines in the borrower’s credit file.                                                                  |
| pub_rec              | 1.34531e+06 | 0.21527603303327858           | 0.0                 | 0.0                 | 0.0                 | 0.0                 | 86.0                |     0.601865    | Number of derogatory public records, such as bankruptcy filings, tax liens, or judgments.                                       |
| revol_bal            | 1.34531e+06 | 16248.114859772097            | 0.0                 | 5943.0              | 11134.0             | 19755.75            | 2904836.0           | 22328.2         | The total amount of credit revolving balances.                                                                                  |
| revol_util           | 1.34531e+06 | 51.77765450920889             | 0.0                 | 33.400001525878906  | 52.099998474121094  | 70.69999694824219   | 892.2999877929688   |    24.5468      | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.      |
| total_acc            | 1.34531e+06 | 24.980838617121705            | 2.0                 | 16.0                | 23.0                | 32.0                | 176.0               |    11.9985      | The total number of credit lines currently in the borrower’s credit file.                                                       |
| mort_acc             | 1.34531e+06 | 1.6120633905939894            | 0.0                 | 0.0                 | 1.0                 | 3.0                 | 51.0                |     1.98892     | Number of mortgage accounts.                                                                                                    |
| pub_rec_bankruptcies | 1.34531e+06 | 0.13437423344805285           | 0.0                 | 0.0                 | 0.0                 | 0.0                 | 12.0                |     0.377843    | Number of public record bankruptcies.                                                                                           |
| issue_d              | 1.34531e+06 | 2015-06-06 17:48:20.835049728 | 2007-06-01 00:00:00 | 2014-07-01 00:00:00 | 2015-08-01 00:00:00 | 2016-07-01 00:00:00 | 2018-12-01 00:00:00 |   nan           | Date the loan was issued                                                                                                        |
| issue_month          | 1.34531e+06 | 6.508846288216099             | 1.0                 | 3.0                 | 7.0                 | 10.0                | 12.0                |     3.44988     | Month of the year the loan was issued                                                                                           |
| issue_year           | 1.34531e+06 | 2014.9717648720368            | 2007.0              | 2014.0              | 2015.0              | 2016.0              | 2018.0              |     1.63969     | Year the loan was issued                                                                                                        |
## Loan Status Distribution
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/01-Loan_Status_Distribution.png)
## Correlation Matrix
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/02-Correlation_Matrix.png)
## Distribution of Installments By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/03-Distribution_of_Installments_By_Loan_Status.png)
## Distribution of Loan Amount By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/04-Distribution_of_Loan_Amount_By_Loan_Status.png)
## Distribution of Interest Rate By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/05-Distribution_of_Interest_Rate_By_Loan_Status.png)
## Distribution of Annual Income By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/06-Distribution_of_Annual_Income_By_Loan_Status.png)
## Violin of Installments By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/07-Violin_of_Installments_By_Loan_Status.png)
## Violin of Loan Amount By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/08-Violin_of_Loan_Amount_By_Loan_Status.png)
## Distribution of Loan Grade By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/09-Distribution_of_Loan_Grade_By_Loan_Status.png)
## Distribution of Loan Sub Grade By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/10-Distribution_of_Loan_Sub_Grade_By_Loan_Status.png)
## Distribution of Home Ownership Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/11-Distribution_of_Home_Ownership_Status_By_Loan_Status.png)
## Distribution of Income Verification Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/12-Distribution_of_Income_Verification_Status_By_Loan_Status.png)
## Distribution of Term Length By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/13-Distribution_of_Term_Length_By_Loan_Status.png)
## Distribution of Loan Purpose By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/14-Distribution_of_Loan_Purpose_By_Loan_Status.png)
## Distribution of Issue Date By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/15-Distribution_of_Issue_Date_By_Loan_Status.png)
## Distribution of Debt To Income By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/16-Distribution_of_Debt_To_Income_By_Loan_Status.png)
## Distribution of Earliest Credit Line Date By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/16-Distribution_of_Earliest_Credit_Line_Date_By_Loan_Status.png)
## Distribution of Debt To Income By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/17-Distribution_of_Debt_To_Income_By_Loan_Status.png)
## Distribution of Number Of Open Accounts By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/17-Distribution_of_Number_Of_Open_Accounts_By_Loan_Status.png)
## Distribution of Number Of Open Accounts By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/18-Distribution_of_Number_Of_Open_Accounts_By_Loan_Status.png)
## Distribution of Revolving Line Utilization Rate By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/18-Distribution_of_Revolving_Line_Utilization_Rate_By_Loan_Status.png)
## Distribution of Revol Util By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/19-Distribution_of_Revol_Util_By_Loan_Status.png)
## Distribution of Revolving Balance By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/19-Distribution_of_Revolving_Balance_By_Loan_Status.png)
## Distribution of Number of Public Records By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/20-Distribution_of_Number_of_Public_Records_By_Loan_Status.png)
## Distribution of Revol Bal By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/20-Distribution_of_Revol_Bal_By_Loan_Status.png)
## Distribution of Initial Listing Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/21-Distribution_of_Initial_Listing_Status_By_Loan_Status.png)
## Distribution of Number of Public Records By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/21-Distribution_of_Number_of_Public_Records_By_Loan_Status.png)
## Distribution of Application Type By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/22-Distribution_of_Application_Type_By_Loan_Status.png)
## Distribution of Initial Listing Status By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/22-Distribution_of_Initial_Listing_Status_By_Loan_Status.png)
## Distribution of Application Type By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/23-Distribution_of_Application_Type_By_Loan_Status.png)
## Distribution of Number of Public Record Bankruptcies By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/23-Distribution_of_Number_of_Public_Record_Bankruptcies_By_Loan_Status.png)
## Correlation Between Loan Status And Numerical Features
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/24-Correlation_Between_Loan_Status_And_Numerical_Features.png)
## Distribution of Number of Public Record Bankruptcies By Loan Status
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/25-Distribution_of_Number_of_Public_Record_Bankruptcies_By_Loan_Status.png)
## Correlation Between Loan Status And Numerical Features
![Alt Text](https://raw.githubusercontent.com/Andrerg01/All_Lending_Club_ML_Analysis/main/outputs/prototype/figures/26-Correlation_Between_Loan_Status_And_Numerical_Features.png)
