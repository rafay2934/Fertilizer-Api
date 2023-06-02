from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
from sklearn import tree
import uvicorn
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}


# Define the input data model
class FertilizerData(BaseModel):
    Temperature: int
    Humidity: int
    Moisture: int
    Soil_Type: int
    Crop_Type: int
    Nitrogen: int
    Potassium:int
    Phosphorus:int


# Load the trained decision tree model
with open('RFModel_Fertilizer.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the API endpoint for GET request
@app.get('/predict_fertilizer')
def get_predict_fertilizer():
    return {'message': 'Send a POST request to this endpoint with fertilizer data.'}


# Define the API endpoint for POST request
@app.post('/predict_fertilizer')
def post_predict_fertilizer(request: Request, data: FertilizerData):
    # Preprocess the input data if necessary
    input_data = [[
        data.Temperature,
        data.Humidity,
        data.Moisture,
        data.Soil_Type,
        data.Crop_Type,
        data.Nitrogen,
        data.Potassium,
        data.Phosphorus

    ]]

    # Make predictions using the loaded model
    predicted_fertilizer = model.predict(input_data)[0]

    # Return the predicted crop as the API response
    return {'fertilizer': predicted_fertilizer}



