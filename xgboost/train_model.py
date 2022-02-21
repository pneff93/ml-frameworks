import argparse

import numpy as np
import xgboost
from xgboost import DMatrix

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", help="number of iterations", type=int, default=1000)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.3)
parser.add_argument("-msl", "--min_split_loss", help="min split loss", type=float, default=0)
parser.add_argument("-d", "--max_depth", help="max depth", type=int, default=6)
parser.add_argument("-model", "--existing_model", help="current model", type=str)
parser.add_argument("-mse", "--mse_threshold", help="MSE threshold", type=float)
parser.add_argument("-obj", "--objective", help="objective", type=str, default="reg:squarederror")
parser.add_argument("-b", "--booster", help="booster", type=str, default="gbtree")


def _calculate_mse(true_values: np.ndarray, predictions: np.ndarray):
    if len(predictions.shape) != 1:
        raise NotImplemented("Only single prediction values are supported.")
    if len(true_values.shape) != 1:
        raise NotImplemented("Only single true values are supported.")
    errors = true_values - predictions
    squared_errors = errors ** 2
    return np.average(squared_errors)


def train_model(training_data: DMatrix, validation_data: DMatrix, mse_threshold=None, existing_model=None,
                objective="reg:squarederror",
                booster="gbtree",
                iterations=1000,
                learning_rate=0.3,
                min_split_loss=0,
                depth=6):
    booster_params = {}
    booster_params.setdefault("objective", objective)
    booster_params.setdefault("booster", booster)
    booster_params.setdefault("learning_rate", learning_rate)
    booster_params.setdefault("min_split_loss", min_split_loss)
    booster_params.setdefault("max_depth", depth)

    print("train model with parameters:", booster_params)

    model = xgboost.train(
        params=booster_params,
        dtrain=training_data,
        num_boost_round=iterations,
        xgb_model=existing_model
    )

    true_values: np.ndarray = validation_data.get_label()
    predicted_values: np.ndarray = model.predict(validation_data)
    mse = _calculate_mse(true_values, predicted_values)
    print("MSE:", mse)

    # logic whether to save (new) model or not
    if existing_model is None:
        print("new model trained")

        if mse_threshold is None:
            print("No threshold for MSE set. Save current model.")
            model.save_model("model.json")
        elif mse_threshold > mse:
            print("MSE is better than threshold. Save current model.")
            model.save_model("model.json")
        else:
            print("MSE is not better than threshold. Discard current model.")

    else:
        print("model is retrained")
        old_model = xgboost.Booster()
        old_model.load_model(existing_model)

        predicted_values_old: np.ndarray = old_model.predict(validation_data)
        mse_old = _calculate_mse(true_values, predicted_values_old)
        if mse < mse_old:
            print("MSE improved from", mse_old, "to", mse, ". Safe new model")
            model.save_model("model.json")
            if mse <= mse_threshold:
                print("MSE threshold fulfilled. No more retraining needed.")
        else:
            print("MSE could not be improved. Keep old model. Current MSE:", mse_old)


if __name__ == "__main__":
    parser.add_argument("-td", "--training_data", help="training data", type=xgboost.DMatrix, required=True)
    parser.add_argument("-vd", "--validation_data", help="validation data", type=xgboost.DMatrix, required=True)
    args = parser.parse_args()
    train_model(args.training_data, args.validation_data, args.mse_threshold, args.existing_model, args.objective,
                args.booster, args.iterations, args.learning_rate, args.min_split_loss, args.depth)
