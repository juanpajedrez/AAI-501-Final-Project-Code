# Credit Card History Fraud Detection
This project is part of the AAI-501 course in the MSAAI Applied Artificial Intelligence Program at the University of San Diego (USD).

** --Project Status: Completed **

### Installation

To run this project on your local machine:

1. Clone the repository
    ```bash
    git clone https://github.com/juanpajedrez/AAI-501-Final-Project-Code 
    ```
2. Run with Jupyter: https://jupyter.org/install 

# Project Intro/Objective:
The main objective of this project is to create a classification model that can accurately predict fraudulent transactions based on numerical and categorical features, such as transaction datetime, merchant, category, first and last names, gender, street name, city, state, job, date of birth, transaction number, amount number, and many more. The model would predict fraud or not based on these features.

### Contributors
- Marquise Oliver
- Joel Dievendorf
- Juan Pablo Triana

### Methods Used
- Data fetching and loading
- Data preprocessing
- EDA and inferential statistics
- Logistic Regression
- Decision tree
- Linear Support Vector Machine
- LSTM model
- Model Evaluation
- Data Visualization

### Technologies
- Python
- Jupyter Notebook

## Project Description

### Dataset
- **Source**: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data?select=credit_card_transactions.csv
- **Variables**: The dataset includes features such as `amount,` `gender,` `date of transaction,` `merchant,` and `is_fraud` (target variable).
- **Size** The dataset contains 1296675 rows with 23 columns

The dataset was cleaned by handling missing values, dropping unique identifiers, encoding low and high categorical features using frequency encoding and target encoding using sci-kit-learn, and using Synthetic Minority Over Sampling (SMOTE) to balance the dataset.

- **Preprocessed Dataset** The dataset contains 1296675 rows with 96 numerical columns.

## Models: Used
- **Logistic Regression** is a probabilistic Binary model that predicts probabilities based on linear relationships between features and the target.
- **Linear Support Vector Machine (SVM)** Nonparametric model that finds the optimal hyperplane or “street” separating data based on dot products between samples.
- **Decision Tree** Tree algorithms that split data hierarchically based on feature thresholds using the difference between nodes with entropy criterion.
- **LSTM (Long Short-Term Memory)** A recurrent neural network (RNN) type designed to capture long-term dependencies in sequential data using cell state memory.

## Project Steps:
1. **Data Cleaning/Preparation**: Missing values, drop unique identifiers, encode time variables, encode categorical features, and handle imbalanced dataset with SMOTE
2. **Exploratory Data Analysis**: The data was visualized using correlation matrices before and after SMOTE to explore relationships between features and binary target variable fraud.
3. **Model Selection**: The following were the cross-validation results: (F1 scores: Decision Trees = 0.9988; Logistic Regression = 0.8969; Linear SVMs = 0.8809; LSTM = 0.9771).
4. **Model Analysis**: It appears the decision trees behaved better with binary, continuous, mixed numerical features, with an F1 score of 0.9988. This was followed by LSTM, then logistic regression, and finally linear SVMs.
5. **Conclusion & Recommendations**: Decision trees were the best model, followed by the LSTM network, to identify fraudulent transactions.
