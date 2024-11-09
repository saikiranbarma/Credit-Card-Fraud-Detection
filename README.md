# Credit-Card-Fraud-Detection

### Project Overview

The Credit Card Fraud Detection project focuses on detecting fraudulent transactions in credit card transaction data. The goal is to build a machine learning model that can identify suspicious activities based on features like transaction time, amount, and anonymized variables. The dataset is highly imbalanced, with fraudulent transactions accounting for only a small percentage of the total, which presents a unique challenge for model training.

#### This project involves the following key steps:

##### Data Preprocessing: 
Handling missing values, scaling features, and preparing the dataset for model training.

##### Exploratory Data Analysis (EDA):
Understanding the dataset by identifying patterns and visualizing data distributions.

##### Feature Engineering:
Creating new features or transforming existing ones to improve model performance.

##### Model Training:
Implementing machine learning algorithms such as XGBoost, Logistic Regression, and Random Forest to detect fraud.

##### Model Evaluation and Comparison: 
Assessing the performance of the models using various evaluation metrics.

##### Model Comparison:
Comparing multiple models to select the most effective one for fraud detection.

### Data

The dataset used for this project is the Credit Card Fraud Detection dataset, which contains over 280,000 credit card transactions. It includes anonymized features, as well as a target variable that indicates whether the transaction was fraudulent (1) or legitimate (0). The data is highly imbalanced, with fraudulent transactions representing only a small fraction of the total.

### Data Features:
##### Time: 
The time (in seconds) since the first transaction in the dataset.
##### Amount: 
The amount of the transaction.
##### Class: 
The target variable, where 1 represents a fraudulent transaction and 0 represents a legitimate one.
##### V1-V28:
Anonymized features representing various characteristics of the transaction.
### Data Loading

The data is loaded into a pandas DataFrame and the target variable (Class) is separated from the features. Since the dataset may have missing values or inconsistencies, it is essential to perform a quick check for such issues before proceeding with the analysis.

Once the dataset is loaded, it is divided into features (X) and the target variable (y), where X contains the anonymized features and y contains the fraud labels.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) involves investigating the dataset to identify patterns, outliers, and relationships between features. Key steps in the EDA process include:

##### Class Distribution:
One of the first things to check is the distribution of the target variable (Class). This allows us to understand the class imbalance, which is a crucial factor for training machine learning models. Fraudulent transactions represent only a small fraction of the total transactions.

##### Correlation Analysis:
By analyzing correlations between features, we can identify any relationships between them that might be important for predicting fraud. A heatmap is often used to visualize correlations between numerical features.

##### Feature Distributions: 
Visualizing the distributions of key features, such as transaction amount and time, helps in understanding their impact on model performance. Some features may require transformations or adjustments to be useful for prediction.

### Feature Engineering

Feature engineering is the process of transforming raw data into a more useful form for model training. Key steps involved in feature engineering include:

##### Scaling: 
The features in the dataset have different scales, which can affect the performance of machine learning models. For example, transaction amounts can vary significantly compared to other features. Standardizing or normalizing these values ensures that the model treats all features equally.
##### Handling Class Imbalance: 
Since fraudulent transactions are rare, the dataset is highly imbalanced. To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied to oversample the minority class (fraudulent transactions) and balance the dataset.
##### Train-Test Split: 
The data is split into training and testing sets, ensuring that both sets contain a representative distribution of the target variable. This allows the model to be trained on one portion of the data and evaluated on another, simulating real-world conditions.
### Machine Learning Model Evaluation

Once the data has been preprocessed and features have been engineered, machine learning models are trained to predict fraudulent transactions. In this project, three models are implemented:

##### Model 1: XGBoost Classifier
XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting algorithm that is particularly effective for handling imbalanced datasets like this one. It is trained on the preprocessed and resampled data, and its performance is evaluated using a set of classification metrics, including precision, recall, F1-score, and ROC-AUC.

##### Model 2: Logistic Regression
Logistic Regression is used as a baseline model. Although less complex than XGBoost, it is a useful model for comparison and provides a baseline performance to assess the effectiveness of more advanced algorithms.

##### Model 3: Random Forest Classifier
Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and control overfitting. It is trained on the resampled data and evaluated similarly to the other models. Random Forests can capture complex interactions between features and are often used for imbalanced datasets like fraud detection.

### Model Evaluation and Comparison

After training the models, they are evaluated using several metrics to assess their ability to detect fraudulent transactions. The most commonly used evaluation metrics in classification tasks are:

##### Accuracy:
The proportion of correct predictions (both fraudulent and legitimate transactions).
##### Precision: 
The percentage of fraudulent transactions correctly identified by the model.
##### Recall: 
The percentage of actual fraudulent transactions that were correctly identified by the model.
##### F1-Score:
The harmonic mean of precision and recall, providing a balance between the two.
##### ROC-AUC:
The area under the Receiver Operating Characteristic curve, which measures the model’s ability to discriminate between fraudulent and legitimate transactions.
## Model Comparison:
After evaluating the models, a comparison is made to select the most effective model. Typically, XGBoost tends to outperform Logistic Regression in terms of recall and F1-score, due to its ability to handle class imbalance. However, Random Forest also performs well in this context and is particularly effective at handling complex interactions between features.

The Random Forest model may provide a good balance of interpretability and performance. For fraud detection, models like XGBoost and Random Forest are often preferred due to their ability to handle imbalanced datasets and capture non-linear relationships in the data.

## Contributors

Saikiran Barma: Project lead, responsible for data preprocessing, feature engineering, model training, and evaluation.
We welcome contributions to improve this project. If you would like to contribute, please follow the steps outlined below.

## How to Contribute

Fork this repository.

Create a new branch for your changes.

Commit your changes with descriptive messages.

Push your changes to your forked repository.

Open a pull request to the main repository.
