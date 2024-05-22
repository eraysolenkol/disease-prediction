import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier


def test_model(model, model_name, x_test, y_test):
    test_accuracy = model.score(x_test.values, y_test)
    print(f'Test Accuracy {model_name}: {test_accuracy}')


def main():
    training_data = pd.read_csv('prediction/core/data/filtered_training_data.csv')
    y = training_data['prognosis']
    x = training_data.drop(columns=['prognosis'])
    x = x.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=42)

    RandomForest(X_train, y_train, X_test, y_test) #1
    DecisionTree(X_train, y_train, X_test, y_test) #2
    NaiveBayesMultinomial(X_train, y_train, X_test, y_test) #3
    SVM(X_train, y_train, X_test, y_test) #4
    KNN(X_train, y_train, X_test, y_test) #5
    RandomForest2(X_train, y_train, X_test, y_test) #6
    DecisionTree2(X_train, y_train, X_test, y_test) #7
    GradientBoosting(X_train, y_train, X_test, y_test) #8

    

def RandomForest2(X_train, y_train, x_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=10)
    rf_model.fit(X_train.values, y_train)
    test_model(rf_model, 'Random Forest2', x_test, y_test)
    return rf_model

def DecisionTree2(X_train, y_train, X_test, y_test):
    decision_tree_model = DecisionTreeClassifier(criterion='entropy', random_state=5, max_depth=10, min_samples_split=12)
    decision_tree_model.fit(X_train.values, y_train)
    test_model(decision_tree_model, 'Decision Tree2', X_test, y_test)
    return decision_tree_model


def NaiveBayesMultinomial(X_train, y_train, X_test, y_test):
    naive_bayes_model = MultinomialNB(alpha=0.01)
    naive_bayes_model.fit(X_train.values, y_train)
    test_model(naive_bayes_model, 'Naive Bayes MultinomialNB', X_test, y_test)

def GradientBoosting(X_train, y_train, X_test, y_test):
    gbm = GradientBoostingClassifier(n_estimators=150,criterion='friedman_mse')
    gbm.fit(X_train.values, y_train)
    test_model(gbm, 'Gradient Boosting Classifier', X_test, y_test)

def DecisionTree(X_train, y_train, X_test, y_test):
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train.values, y_train)
    test_model(decision_tree_model, 'Decision Tree', X_test, y_test)

def NaiveBayes(X_train, y_train, X_test, y_test):
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train.values, y_train)
    test_model(naive_bayes_model, 'Naive Bayes GaussianNB', X_test, y_test)

def SVM(X_train, y_train, X_test, y_test):
    svm_model = SVC(kernel='linear', C=1, probability=True)
    svm_model.fit(X_train.values, y_train)
    test_model(svm_model, 'SVM', X_test, y_test)

def KNN(X_train, y_train, X_test, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train.values, y_train)
    test_model(knn_model, 'KNN', X_test, y_test)

def RandomForest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(random_state=42, n_estimators=500)
    rf_model.fit(X_train.values, y_train)
    test_model(rf_model, 'Random Forest', X_test, y_test)


main()