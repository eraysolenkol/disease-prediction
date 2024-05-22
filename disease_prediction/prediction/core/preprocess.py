import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def entropy(y):
    # entropy değerini hesaplar
    counts = np.bincount(y)
    probabilities = counts / len(y)
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy_value

def split_dataset(X, y, attribute, threshold):
    # Dataseti belirli bir attribute ve threshold değerine göre ikiye böler
    left_mask = X[:, attribute] <= threshold
    right_mask = X[:, attribute] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def information_gain(X, y, attribute, threshold):
    # Information gaini hesaplama
    parent_entropy = entropy(y)
    
    X_left, y_left, X_right, y_right = split_dataset(X, y, attribute, threshold)
    
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    child_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    
    return parent_entropy - child_entropy

# Dataseti yükleyip X'in son sütunu boş olduğu için onu siliyoruz
training_data = pd.read_csv('prediction/core/data/training_data.csv')
y = training_data['prognosis']
X = training_data.drop(columns=['prognosis'])
X = X.iloc[:, :-1]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lower_threshold = 0.15
upper_threshold = 0.50

threshold_value = 0.5

attributes_to_delete = []

# Information gain aralıkta değilse o attribute'u siliyoruz (semptomu)
for attribute_index in range(X_scaled.shape[1]):
    ig = information_gain(X_scaled, y_encoded, attribute_index, threshold_value)
    if ig < lower_threshold or ig > upper_threshold:
        attributes_to_delete.append(attribute_index)

test = pd.read_csv('prediction/core/data/test_data.csv')

if attributes_to_delete:
    for attribute_index in attributes_to_delete:
        symptom_column_name = X.columns[attribute_index]
        del training_data[symptom_column_name]
        del test[symptom_column_name]
    

    training_data.to_csv('prediction/core/data/filtered_training_data.csv', index=False)
    test.to_csv('prediction/core/data/filtered_test_data.csv', index=False)
