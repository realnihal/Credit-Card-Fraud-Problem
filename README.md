# Credit Card Fraud Kaggle


## Context and Dataset information

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The dataset that we are using is obtained from the [**Kaggle dataset by the Machine Learning Group - ULB**](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The dataset that we are using contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. **This dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions**.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, they cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. **Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise**.

[**To check out the complete code for the project click here**](https://github.com/realnihal/Credit-Card-Fraud-Problem).

```python
df.Class.value_counts()
```
    0    284315
    1       492
    Name: Class, dtype: int64


This is interesting! the dataset is imbalanced. 
In classification machine learning problems(binary and multiclass), datasets are often imbalanced which means that one class has a higher number of samples than others. This will lead to bias during the training of the model, the class containing a higher number of samples will be preferred more over the classes containing a lower number of samples. Having bias will, in turn, increase the true-negative and false-positive rates (ie, the precision and recall).

Let's see the results without adjusting for the imbalanced bias on a base-model. I have used a **simple logistic regression model** for this.

The models that we made and their results are as follows:

##  **1. Base-Model**

```
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.77      0.68      0.72       344
    
        accuracy                           1.00    199365
       macro avg       0.89      0.84      0.86    199365
    weighted avg       1.00      1.00      1.00    199365
```  


## **2. RandomForestClassifier**

```
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.94      0.73      0.82       344
    
        accuracy                           1.00    199365
       macro avg       0.97      0.86      0.91    199365
    weighted avg       1.00      1.00      1.00    199365
```    

## **3. Under-Sampling**

                  precision    recall  f1-score   support
    
               0       1.00      0.89      0.94    199021
               1       0.01      0.92      0.03       344
    
        accuracy                           0.89    199365
       macro avg       0.51      0.91      0.49    199365
    weighted avg       1.00      0.89      0.94    199365
    

## **4. Over-Sampling**


                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.94      0.76      0.84       344
    
        accuracy                           1.00    199365
       macro avg       0.97      0.88      0.92    199365
    weighted avg       1.00      1.00      1.00    199365


## **5. SMOTETomek**


                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.86      0.82      0.84       344
    
        accuracy                           1.00    199365
       macro avg       0.93      0.91      0.92    199365
    weighted avg       1.00      1.00      1.00    199365
    
## **6. SMOTE**

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.87      0.84      0.85       344
    
        accuracy                           1.00    199365
       macro avg       0.93      0.92      0.93    199365
    weighted avg       1.00      1.00      1.00    199365

## **7. Extra-Trees Classifier**

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.87      0.84      0.85       344
    
        accuracy                           1.00    199365
       macro avg       0.93      0.92      0.93    199365
    weighted avg       1.00      1.00      1.00    199365
    
  

## Conclusions

We have tried to solve our problem of data imbalance using multiple approaches. The best model that we could produce was between the extra-trees and the SMOTE model. Further attempts could be made with other models.

To check out the article for this project [**click here**](https://github.com/realnihal/Credit-Card-Fraud-Problem).

And with that Peace out!
