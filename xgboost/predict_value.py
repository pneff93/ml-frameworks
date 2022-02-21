import argparse
import pandas as pd
import xgboost
from xgboost import DMatrix


def predict_value(
        trip_seconds: str,
        trip_miles: str,
        pickup_community_area: str,
        dropoff_community_area: str,
        fare: str,
        tolls: str,
        extras: str,
        trip_total: str):

    # load model
    model = xgboost.Booster()
    model.load_model("model.json")

    # create data frame
    data = [
        {"trip_seconds": trip_seconds,
         "trip_miles": trip_miles,
         "pickup_community_area": pickup_community_area,
         "dropoff_community_area": dropoff_community_area,
         "fare": fare,
         "tolls": tolls,
         "extras": extras,
         "trip_total": trip_total}
    ]

    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='coerce')

    test_data: DMatrix = xgboost.DMatrix(
        data=df
    )

    prediction = model.predict(test_data)
    print(prediction)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ts", "--trip_seconds", help="trip seconds")
    parser.add_argument("-tm", "--trip_miles", help="trip miles")
    parser.add_argument("-p", "--pickup_community_area", help="pickup community area")
    parser.add_argument("-d", "--dropoff_community_area", help="dropoff community area")
    parser.add_argument("-f", "--fare", help="fare")
    parser.add_argument("-t", "--tolls", help="tolls")
    parser.add_argument("-e", "--extras", help="extras")
    parser.add_argument("-tt", "--trip_total", help="trip total")
    args = parser.parse_args()
    predict_value(args.trip_seconds, args.trip_miles, args.pickup_community_area, args.dropoff_community_area,
                  args.fare, args.tolls, args.extras, args.trip_total)
