# pip freeze | grep scikit-learn
# pipenv install scikit-learn==1.1.2 flask --python=3.9.13

import pickle
from flask import Flask, request, jsonify


app = Flask("duration-prediction")

with open("lin_reg.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    preds = predict(features) 
    result = {
        "duration": preds
        }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

#linux
# for production ready: pip install gunicorn
# gunicorn --bind=0.0.0.0:9696 predict:app

#windows
# for production ready: pip install waitress
# waitress-serve --listen=0.0.0.0:9696 predict:app
