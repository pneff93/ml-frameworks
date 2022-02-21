import json

from flask import Flask, jsonify, request
import predict

app = Flask(__name__)


@app.route("/prediction", methods=['POST'])
def get_prediction():
    data = json.loads(request.data)
    review = str(data["review"])
    prediction, classification = predict.predict(review)

    return jsonify(
        review=review,
        prediction=round(float(prediction), 2),
        rating=classification
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088)
