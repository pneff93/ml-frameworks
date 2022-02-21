from flask import Flask, jsonify, request
import predict_value

app = Flask(__name__)


@app.route("/prediction", methods=['GET'])
def get_prediction():
    trip_seconds = request.args.get('trip_seconds')
    trip_miles = request.args.get('trip_miles')
    pickup_community_area = request.args.get('pickup_community_area')
    dropoff_community_area = request.args.get('dropoff_community_area')
    fare = request.args.get('fare')
    tolls = request.args.get('tolls')
    extras = request.args.get('extras')
    trip_total = request.args.get('trip_total')

    prediction = predict_value.predict_value(trip_seconds,
                                             trip_miles,
                                             pickup_community_area,
                                             dropoff_community_area,
                                             fare,
                                             tolls,
                                             extras,
                                             trip_total)

    return str(prediction[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088)
