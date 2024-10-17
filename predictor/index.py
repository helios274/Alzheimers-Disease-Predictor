import pickle
import numpy as np


features = ['Age', 'BMI', 'AlcoholConsumption', 'CholesterolTotal',
            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
            'PhysicalActivity', 'DietQuality', 'SleepQuality', 'MMSE',
            'FunctionalAssessment', 'ADL', 'MemoryComplaints', 'BehavioralProblems']


with open('predictor/models/model_gbc_v2-3.pkl', 'rb') as file:
    model = pickle.load(file)


def make_prediction(data):

    input_data = [data.get(f) for f in features]

    input_data = np.array(input_data).reshape(1, -1)

    proba = model.predict_proba(input_data)[0].tolist()[1]
    proba = round(proba*100, 2)

    return proba
