import time
import argparse

from api import dump_to_json, DataSud, Opendatasoft, DataGouv


def get_open_data_soft():
    start = time.time()
    open_data_soft = Opendatasoft()
    datasets = open_data_soft.get_data()
    end = time.time()
    dump_to_json("data/opendatasoft.json", datasets)
    print(f"Fetched from Opendatasoft in {end - start}s")


def get_data_sud():
    start = time.time()
    data_sud = DataSud()
    datasets = data_sud.get_data()
    end = time.time()
    dump_to_json("data/datasud.json", datasets)
    print(f"Fetched from DataSud in {end - start}s")


def get_data_gouv():
    start = time.time()
    data_gouv = DataGouv()
    datasets = data_gouv.get_data()
    end = time.time()
    dump_to_json("data/datagouv.json", datasets)
    print(f"Fetched from DataGouv in {end - start}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch data from APIs")
    parser.add_argument(
        "-o",
        "--open-data-soft",
        help="Fetch from Open Data Soft",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-s",
        "--data-sud",
        help="Fetch from Data Sud",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-g",
        "--data-gouv",
        help="Fetch from Data Gouv",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    if args.open_data_soft:
        get_open_data_soft()
    if args.data_sud:
        get_data_sud()
    if args.data_gouv:
        get_data_gouv()
