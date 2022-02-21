import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="filename", type=str, required=True)


def read_data(path: str):
    try:
        df: pd.DataFrame = pd.read_csv(
            path,
        )
        print("read data successful")
        print("columns with NaN: ", df.columns[df.isna().any()].tolist())
        return df

    except FileNotFoundError:
        print(path, "not found.")


if __name__ == "__main__":
    args = parser.parse_args()
    read_data(args.file)
