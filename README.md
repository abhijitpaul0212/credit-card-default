---
license: mit
title: Credit Card Defaults Prediction
sdk: streamlit
emoji: 🦀
colorFrom: green
colorTo: indigo
app_file: streamlit_app.py
pinned: false
---

# Credit Card Default Detection

## Objective
This project focuses on building a credit card fault detection model using machine learning techniques. The model aims to predict default payments based on various demographic and credit-related features.

## Dataset
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. 

For more information on the dataset, please visit the UCI ML Repository
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

or Kaggle website at https://www.kaggle.com/code/selener/prediction-of-credit-card-default/input

## Machine Learning Pipeline
1. Analyze Data: In this initial step, we attempted to comprehend the data and searched for various available features. We looked for things like the shape of the data, the data types of each feature, a statistical summary, etc. at this stage.
2. EDA: EDA stands for Exploratory Data Analysis. It is a process of analyzing and understanding the data. The goal of EDA is to gain insights into the data, identify patterns, and discover relationships and trends. It helps to identify outliers, missing values, and any other issues that may affect the analysis and modeling of the data.
3. Data Cleaning: Data cleaning is the process of identifying and correcting or removing inaccuracies, inconsistencies, and handling missing values in a dataset. We inspected the dataset for duplicate values. The null value and outlier detection and treatment followed. For the imputation of the null value we used the Mean, Median, and Mode techniques, and for the outliers, we used the Clipping method to handle the outliers without any loss to the data.
4. Feature Selection: At this step, we did the encoding of categorical features. We used the correlation coefficient, encoding, feature manipulation, and feature selection techniques to select the most relevant features. SMOTE is used to address the class imbalance in the target variable.
5. Feature Scaling: We scaled the features to bring down all of the values to a similar range. 
6. Model Selection and Implementation: We pass the features to SVM, KNN, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoosting, Naive Bayes & XGBoost classification algorithms. We also did hyperparameter tuning using GridSearchCV.
7. Performance Evaluation: After passing it to various classification models and calculating the metrics viz. accuracy, precision, recall, f1, roc-auc,  we choose a final model that can make best predictions.

## Artifacts

#### Dataset Source
MongoDB

#### Preprocessings steps
1. Handling Outliers
2. Scaling data
3. Handling imbalance dataset


#### Algorithms used to find best model
1. LogisticRegression
2. SVC
3. RandomForestClassifier
4. GradientBoostingClassifier
5. KNeighborsClassifier
6. DecisionTreeClassifier
7. XGBoost

## Final Result
The XGBClassifier model emerged as the most effective have below metric scores:
* Accuracy: 83%
* Recall: 80%
* ROC-AUC: 91%
* Precision: 84%
* F1: 81%

## Deployed URLs
* STREAMLIT: https://credit-card-defaults-prediction.streamlit.app/
* HUGGINGFACE: https://huggingface.co/spaces/abhijitpaul/credit-card-default
* AWS: https://5hniewmhgh.us-east-1.awsapprunner.com/

## Project Artifacts
1. High Level Design (HLD): [HLD_CreditCardDefaultDetection.pdf](https://drive.google.com/file/d/10OCGzx_PPRrzk0ZGTjAu-N-3M30baJbr/view?usp=sharing)
2. Low Level Design (LLD): [LLD_CreditCardDefaultDetection.pdf](https://drive.google.com/file/d/1i7ZKx161WQdlWNsOhWOK2Nr4DkSG7mIV/view?usp=sharing)
3. Architecture Design: [Architechture_CreditCardDefaultDetection.pdf](https://drive.google.com/file/d/11ZwHiFd5yPq4XG6MASN9GLnAVc4DwWeQ/view?usp=sharing)
4. Wireframe Document: [Wireframe_CreditCardDefaultDetection.pdf](https://drive.google.com/file/d/1f4dd32-QLc6mEmI-LOpKGNpsZpF0nbJ9/view?usp=sharing)
5. Detailed Project Report (DPR): [DPR_CreditCardDefaultDetection.pdf](https://docs.google.com/presentation/d/1mIkaHOO4tTHoA7D1-tLJUH5X57f642ht/edit?usp=sharing&ouid=113072205406904395232&rtpof=true&sd=true)
6. Project Demo Video: [https://youtu.be/8lv5jNUXcKQ?si=Ff1MIxzQMp9MBvj1](https://youtu.be/kH6FAdevtmw)

## Snaptshot of HuggingFace Dashboard
![CreditCardDefaultDetectionDashboard](https://github.com/abhijitpaul0212/Credit-Card-Default-Detection/assets/9966441/da9c2642-6a99-47bf-b517-55ea505eb588)

## Contributors
* Abhijit Paul
* Gouthami K

## License
This project is licensed under the MIT License.
