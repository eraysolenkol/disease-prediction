import joblib as jb
import numpy as np
import json
import pandas as pd


def get_diseases_translations():
    json_file = open('prediction/core/data/diseases_translation.json', encoding='utf-8')
    diseases_translations = json.load(json_file)
    return diseases_translations

def get_precautions():
    csv_file = open('prediction/core/data/symptom_precaution_tr.csv', encoding='utf-8')
    precautions = csv_file.readlines()
    precautions = [precaution.strip() for precaution in precautions]
    precautions.remove('Disease,Precaution_1,Precaution_2,Precaution_3,Precaution_4')
    return precautions

def get_description():
    csv_file = open('prediction/core/data/symptom_description_tr.csv', encoding='utf-8')
    descriptions = csv_file.readlines()
    descriptions = [description.strip() for description in descriptions]
    descriptions.remove('Disease,Description')
    return descriptions

def get_symptoms(lang='en'):
    if lang == 'en':
        with open('prediction/core/data/symptoms.txt', 'r') as file:
            symptoms_string = file.read()
    elif lang == 'tr':
        with open('prediction/core/data/symptoms_tr.txt', 'r', encoding='utf-8') as file:
            symptoms_string = file.read()


    symptomslist = symptoms_string.split(',')
    symptomslist = [symptom.strip() for symptom in symptomslist]
    return symptomslist

models = ['random_forest_model.pkl', 'decision_tree_model.pkl', 'naive_bayes_model.pkl', 'svm_model.pkl', 'knn_model.pkl', 'unknown_model.pkl']

def get_top3_diseases(psymptoms, model):

    if model == 'random_forest':
        model = jb.load(f'prediction/core/models/{models[0]}')
    elif model == 'decision_tree':
        model = jb.load(f'prediction/core/models/{models[1]}')
    elif model == 'naive_bayes':
        model = jb.load(f'prediction/core/models/naive_bayes_model11.pkl')
    elif model == 'svm':
        model = jb.load(f'prediction/core/models/{models[3]}')
    elif model == 'knn':
        model = jb.load(f'prediction/core/models/{models[4]}')
    elif model == 'unknown':
        model = jb.load(f'prediction/core/models/{models[5]}')
    elif model == 'random_forest2':
        model = jb.load(f'prediction/core/models/random_forest_model2.pkl')
    elif model == 'decision_tree2':
        model = jb.load(f'prediction/core/models/decision_tree_model2.pkl')
    elif model == 'gradient_boosting':
        model = jb.load(f'prediction/core/models/gradient_boosting_model.pkl')

    testingsymptoms = [0] * len(symptomslist)

    for k in range(len(symptomslist)):
        for z in psymptoms:
            if z == symptomslist[k]:
                testingsymptoms[k] = 1


    inputtest = [testingsymptoms]

    y_pred_2 = model.predict_proba(inputtest)

    top_three_indices = np.argsort(y_pred_2[0])[::-1][:3]
    top_three_confidences = [y_pred_2[0][i] * 100 for i in top_three_indices]


    diseases_list = []
    for i, confidence in enumerate(top_three_confidences, start=1):
        diseases_list.append({"disease": model.classes_[top_three_indices[i-1]], "confidence": confidence})
    return diseases_list



       

symptomslist = get_symptoms()


def get_disease_symptoms_dict():
    data = pd.read_csv('prediction/core/data/test_data.csv')
    symptom_columns = data.columns[:-1]
    disease_symptoms_dict = {}

    for _, row in data.iterrows():
        disease = row['prognosis']
    
        symptoms = [symptom for symptom in symptom_columns if row[symptom] == 1]
    
        disease_symptoms_dict[disease] = symptoms

    return disease_symptoms_dict

