import argparse

import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from xgboost import DMatrix

parser = argparse.ArgumentParser()
parser.add_argument("-lc", "--label_column", help="label column", type=int, required=True)
parser.add_argument("-sp", "--split_proportion", help="split proportion", type=float, default=0.25)


def preprocess_data(data: pd.DataFrame, label_column: int, split_proportion=0.25):
    if label_column > len(data.columns):
        raise IndexError("Label column is out of scope.")

    if split_proportion >= 1 or split_proportion <= 0:
        raise ValueError("Split proportion should be between (0, 1)")

    training, validation = train_test_split(data, test_size=split_proportion, random_state=1)

    training_data: DMatrix = xgboost.DMatrix(
        data=training.drop(columns=[training.columns[label_column]]),
        label=training[training.columns[label_column]],
    )
    validation_data: DMatrix = xgboost.DMatrix(
        data=validation.drop(columns=[validation.columns[label_column]]),
        label=validation[validation.columns[label_column]],
    )

    print("pre processing was successful. "
          "Training data contains", len(training), "elements.",
          "Validation data contains", len(validation), "elements.")
    return training_data, validation_data


if __name__ == "__main__":
    parser.add_argument("-d", "--data_frame", help="data frame", type=pd.DataFrame, required=True)
    args = parser.parse_args()
    preprocess_data(args.data_frame, args.label_column, args.split_proportion)
