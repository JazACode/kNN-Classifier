import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Jazwur Ankrah
# 001027898
# Machine Learning Assignment 2

def load_csv(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # split the data into features and target variable
    X = data.iloc[:, :-1]  # all columns except the last one
    y = data.iloc[:, -1]   

    # 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    # normalize the data
    X_train_normalized = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test_normalized = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    
    return X_train_normalized, X_test_normalized

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def knn_classify(training_data, labels, test_point, k):
    distances = []
    for i, data_point in enumerate(training_data):
        distance = euclidean_distance(test_point, data_point)
        distances.append((distance, labels[i]))
    
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    
    return most_common_label


# Main function
if __name__ == "__main__":
    
    # load data from csv file
    training_df, testing_df, training_labels, testing_labels = load_csv('wdbc.data.mb.csv')
    
    # normalize the data
    training_normalized, testing_normalized = normalize_data(training_df, testing_df)
    
    # K value array
    k_values = [1, 3, 5, 7, 9]
    
    # loop through the k values
    for k in k_values:
        # initialize confusion matrix
        confusion_matrix = pd.DataFrame(0, index=np.unique(testing_labels), columns=np.unique(testing_labels))
        accuracy = 0
        
        # loop through the test data
        for i, test_point in enumerate(testing_normalized.values):
            predicted_label = knn_classify(training_normalized.values, training_labels.values, test_point, k)
            actual_label = testing_labels.values[i]
            
            # update confusion matrix and accuracy
            confusion_matrix.loc[actual_label, predicted_label] += 1
            if predicted_label == actual_label:
                accuracy += 1
        
        print(f"Confusion matrix for k = {k}:\n{confusion_matrix}\n")
        accuracy /= len(testing_labels)
        print(f"Accuracy for k = {k}: {accuracy:.2f}")
        
    