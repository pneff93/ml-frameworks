import argparse

import pandas as pd

import preprocess_data
import read_data
import train_model


def main(file: str, label_column: int, split_proportion, mse_threshold=None, existing_model=None,
         iterations=1000,
         learning_rate=0.3,
         min_split_loss=0,
         depth=6,
         objective="reg:squarederror",
         booster="gbtree",
         ):
    data: pd.DataFrame = read_data.read_data(file)
    training_data, validation_data = preprocess_data.preprocess_data(data, label_column, split_proportion)
    train_model.train_model(training_data, validation_data, mse_threshold, existing_model, objective,
                            booster, iterations, learning_rate, min_split_loss, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve",
                                     parents=[read_data.parser, preprocess_data.parser, train_model.parser])

    args = parser.parse_args()
    main(args.file, args.label_column, args.split_proportion, args.mse_threshold, args.existing_model,
         args.iterations, args.learning_rate, args.min_split_loss, args.max_depth,
         args.objective, args.booster)
