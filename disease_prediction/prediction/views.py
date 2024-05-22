from django.shortcuts import render
import json
from prediction.core.predict import get_top3_diseases, get_symptoms, get_diseases_translations, get_precautions, get_description, get_disease_symptoms_dict
from django.http import HttpResponse
import logging
import threading
import sqlite3




modelsAndNumbers = {
        "1": "random_forest",
        "2": "decision_tree",
        "3": "naive_bayes",
        "4": "svm",
        "5": "knn",
        "6": "random_forest2",
        "7": "decision_tree2",
        "8": "gradient_boosting"
}



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename="logs/logs.log", format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")

# Create your views here.
def index_admin(request):
    if request.method == "POST":
        logger.info("POST request received for getting the result page")
        body = json.loads(request.body)
        symptoms = body["symptoms"]
        model = body["model"]

        urlToGo = f"/prediction/result/?model={model}&symptoms=" + ",".join(symptoms)
        return HttpResponse(json.dumps({"url": urlToGo}), content_type="application/json")

    logger.info("GET request received for getting the prediction page")
    symptoms = get_symptoms()
    symptoms_tr = get_symptoms('tr')
    disease_symptoms = get_disease_symptoms_dict()

    data = {"symptoms": symptoms, "symptoms_tr": symptoms_tr, "admin": True , "disease_symptoms": disease_symptoms}
    return render(request, "prediction.html", data)

def index(request):
    if request.method == "POST":
        logger.info("POST request received for getting the result page")
        body = json.loads(request.body)
        symptoms = body["symptoms"]
        model = body["model"]
        urlToGo = f"/prediction/result/?model={model}&symptoms=" + ",".join(symptoms)
        return HttpResponse(json.dumps({"url": urlToGo}), content_type="application/json")

    disease_symptoms = get_disease_symptoms_dict()
    logger.info("GET request received for getting the prediction page")
    symptoms = get_symptoms()
    symptoms_tr = get_symptoms('tr')
    data = {"symptoms": symptoms, "symptoms_tr": symptoms_tr, "admin": False, "disease_symptoms": disease_symptoms}
    return render(request, "prediction.html", data)

def result(request):
    con = sqlite3.connect("db.sqlite3", check_same_thread=False)
    cur = con.cursor()
    logger.info("GET request received for getting the result page")
    symptoms = request.GET.get("symptoms")
    symptoms = symptoms.split(",")
    diseases_translations = get_diseases_translations()
    precautions = get_precautions()
    descriptions = get_description()
    model = request.GET.get("model")
    disease_symptoms = get_disease_symptoms_dict()
    symptoms_tr = get_symptoms('tr')
    symptoms_en = get_symptoms()

    diseases = None
    if model == "1":
        diseases = get_top3_diseases(symptoms, model="random_forest")
    elif model == "2":
        diseases = get_top3_diseases(symptoms, model="decision_tree")
    elif model == "3":
        diseases = get_top3_diseases(symptoms, model="naive_bayes")
    elif model == "4":
        diseases = get_top3_diseases(symptoms, model="svm")
    elif model == "5":
        diseases = get_top3_diseases(symptoms, model="knn")
    elif model == "6":
        diseases = get_top3_diseases(symptoms, model="random_forest2")
    elif model == "7":
        diseases = get_top3_diseases(symptoms, model="decision_tree2")
    elif model == "8":
        diseases = get_top3_diseases(symptoms, model="gradient_boosting")
    else:
        diseases = get_top3_diseases(symptoms, model="naive_bayes")
    data = {"diseases": diseases, "diseases_translations": diseases_translations, "precautions": precautions, "descriptions": descriptions, "disease_symptoms": disease_symptoms, "symptoms_tr": symptoms_tr, "symptoms": symptoms_en}
    disease = diseases[0]['disease']
    confidence = diseases[0]['confidence'].round(4)
    symptomsString = ", ".join(symptoms)
    try:
        thread = threading.Thread(target=save_prediction, args=(disease, confidence, symptomsString, modelsAndNumbers[model]))
        thread.start()
    except Exception as e:
        logging.error("ERROR: " + str(e))

    return render(request, "result.html", data)

def save_prediction(disease, confidence, symptoms, model):
    con = sqlite3.connect("db.sqlite3", check_same_thread=False)
    cur = con.cursor()
    try:
        cur.execute("INSERT INTO predictions (disease, confidence, symptoms, model) VALUES (?, ?, ?, ?)", (disease, confidence, symptoms, model))
        con.commit()
    except Exception as e:
        logging.error("ERROR: " + str(e))
    finally:
        cur.close()
        con.close()

def prediction_history(request):
    con = sqlite3.connect("db.sqlite3", check_same_thread=False)
    cur = con.cursor()
    try:
        cur.execute("SELECT disease, confidence, symptoms, model FROM predictions")
        predictions = cur.fetchall()
    except Exception as e:
        logging.error(f"Error fetching prediction history: {e}")
        predictions = []
    finally:
        cur.close()
        con.close()

    data = {"predictions": predictions}
    return render(request, "prediction_history.html", data)