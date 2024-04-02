
# Diabetes Prediction System

## Overview
This repository contains a Python-based system that predicts whether a person has diabetes or not. We utilize a **Support Vector Machine (SVM)** model for this purpose. The system processes relevant health features and classifies individuals as either "diabetic" or "non-diabetic."

## Files
- `notebook.ipynb`: A Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `diabetes.csv`: The dataset used for training and testing the model.

## Dependencies
Make sure you have the following Python libraries installed:
- pandas
- numpy
- scikit-learn



## Usage
1. **Load the Dataset**: The dataset (`diabetes.csv`) contains labeled health features (e.g., glucose level, blood pressure, BMI).
2. **Data Preprocessing**:
    - Clean the data by handling missing values and scaling features.
    - Split the data into features (X) and target labels (Y).
3. **Split the Data**:
    - Divide the dataset into training and testing sets (e.g., 80% training, 20% testing).
4. **Train the Model**:
    - Use SVM to train the model on the preprocessed data.
5. **Evaluate the Model**:
    - Assess the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Results
The model achieved an accuracy of approximately 80% on the training data and 75% on the test data. You can further fine-tune hyperparameters or explore other models to improve performance.

## Future Enhancements
Consider the following improvements:
- **Feature Selection**: Experiment with different subsets of features.
- **Hyperparameter Tuning**: Optimize SVM hyperparameters.
- **Ensemble Methods**: Explore ensemble techniques (e.g., Random Forest, Gradient Boosting).
