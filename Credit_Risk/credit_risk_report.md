# Module 20 Report

## Overview of the Analysis

The purpose of this analysis is to predict the probability of a healthy loan based on input features: loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt. 
The objective is to evaluate the most influencal feature on the loan health.

The stages for implementing a logistic regression model involve:

1. Data Preprocessing:
- Data Collection, Data Cleaning: Handle missing values, outliers, and any data inconsistencies. No need in this specific case as data is fairly clean.
- Feature Selection
- Feature Scaling (was not performed in this challenge)

2. Train-Test Split:
Split the dataset into two parts: a training set and a testing (or validation) set. The training set will be used to train the model, while the testing set will be used to evaluate its performance.

3. Model Construction:
- Define the logistic regression model: Specify the mathematical representation of the model, which involves combining the input features with their corresponding weights and a bias term.
- Activation Function: Use a sigmoid function (also known as the logistic function) to transform the output of the linear equation into a probability value between 0 and 1.

4. Model Training:
Optimization Algorithm: Select an optimization algorithm (e.g., Gradient Descent, Stochastic Gradient Descent) to minimize the cost or loss function. The common loss function for logistic regression is the binary cross-entropy loss.
Update Weights: During training, adjust the weights and bias iteratively to find the optimal values that minimize the loss function.

5. Model Evaluation:
Once the model is trained, use the testing set to evaluate its performance. Common evaluation metrics for binary classification include accuracy, precision, recall, F1 score, and ROC-AUC.
Fine-Tuning: If the model's performance is not satisfactory, consider adjusting hyperparameters, feature selection, or employing more advanced techniques like regularization to improve the model's generalization.
Prediction and Deployment:

Once the model meets the desired performance level, it can be used to make predictions on new, unseen data in real-world applications.
Deploy the model in production environments to perform real-time predictions or integrate it into other systems as required.

Inn the present case, the value count for loan status showed: 
- loan status 0 has 75036 rows 
- loan status 1 is only 2500.

The healthy loan is over represented in the provided data set.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

Training Data Score: 0.9921970722899336
Testing Data Score: 0.9921327543957698

Confusion matrix:
                      Actual        Actual
                      Positive     Negative
Predicted Positive     44776,        233
Predicted Negative      133,        1380

Classification report for the model
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     45009
           1       0.86      0.91      0.88      1513
    accuracy                           0.99     46522
   macro avg       0.93      0.95      0.94     46522
weighted avg       0.99      0.99      0.99     46522

In conclusion, the classification report shows that the model has very high precision, recall, and F1-score for class 0 (label 0), indicating excellent performance in identifying class 0 instances. The model also performs well for class 1 (label 1), though not as perfectly as for class 0. This might be due to the over representation of unhealthy loan values tah we hope to fix in the following analysis using resammpling.

Random Oversampling is a technique used in the context of imbalanced datasets to address the issue of having significantly unequal class distributions. In imbalanced datasets, one class (the minority class) has much fewer instances compared to another class (the majority class). This imbalance can negatively impact the performance of machine learning models, as they tend to be biased towards the majority class and may struggle to properly learn patterns from the minority class.

Random Oversampling involves increasing the number of instances in the minority class by randomly duplicating some of its samples until it reaches a desired balance with the majority class. This is done to ensure that both classes have a more equal representation in the dataset, allowing the model to learn from the minority class more effectively.

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
After resampling the value counts for healthy and unhealthy loan is the same.

balanced_accuracy_score hasnt changed comparted to model 1.
0.9944346079744909

Confusion matrix shows a better results for the unhealthy loan (1)
                      Actual        Actual
                      Positive     Negative
Predicted Positive     44746,        263
Predicted Negative      8,          1505

Classification report for  model 2:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     45009
           1       0.85      0.99      0.92      1513
    accuracy                           0.99     46522
   macro avg       0.93      0.99      0.96     46522
weighted avg       0.99      0.99      0.99     46522
It shows a higher precision and recall and f1 score for the unhealthy loan (1)

# Conclusion 
After Random Oversampling, the model is trained on the balanced dataset, and during the training process, it has equal exposure to both classes. This helped the model to be more sensitive to the patterns in the minority class (1) and improve its performance in correctly classifying both classes.

Model 2 seems more appropriate for the analysis of this data set.

# Bonus:
I have sclaed the data and the results dont show some improvements:
- in the confusion matrix with False Negatives dropping from 133 to 25 for the rescaled model.
- slight though for f1 score and recall

Confusion matrix shows a better results for the unhealthy loan (1)
                          Rescaled Only           Rescaled and reSampled
                      Actual        Actual      Actual        Actual 
                      Positive     Negative     Positive     Negative
Predicted Positive     44756,        253          44719         290
Predicted Negative      25,          1488           8           1505

Classification report for  model 2:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     45009
           1       0.85      0.98      0.91      1513

    accuracy                           0.99     46522
   macro avg       0.93      0.99      0.96     46522
weighted avg       0.99      0.99      0.99     46522

Classification report for  model:
 Rescaled and reSampled
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     45009
           1       0.84      0.99      0.91      1513

    accuracy                           0.99     46522
   macro avg       0.92      0.99      0.95     46522
weighted avg       0.99      0.99      0.99     46522
