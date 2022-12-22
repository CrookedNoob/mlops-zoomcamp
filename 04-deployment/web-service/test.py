import requests

url = "http://localhost:9696/predict"

ride = {
    "PULocationID": 10,
    "DOLocationID": 55,
    "trip_distance": 40
}

#features = predict.prepare_features(ride)
#pred = predict.predict(features)
#print(pred)

response = requests.post(url, json=ride)
print(response.json())