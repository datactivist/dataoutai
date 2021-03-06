import argparse
import time

from api import DataSud, Opendatasoft, DataGouv
from tools import dump_to_json


def get_open_data_soft():
    start = time.time()
    open_data_soft = Opendatasoft()
    datasets = open_data_soft.get_data()
    end = time.time()
    dump_to_json("data/opendatasoft.json", datasets)
    print(
        f"Fetched from Opendatasoft in {end - start}s with {open_data_soft.request_count} requests"
    )


def get_data_sud():
    start = time.time()
    data_sud = DataSud()
    datasets = data_sud.get_data()
    end = time.time()
    dump_to_json("data/datasud.json", datasets)
    print(
        f"Fetched from DataSud in {end - start}s with {data_sud.request_count} requests"
    )


def get_data_gouv():
    start = time.time()
    data_gouv = DataGouv()
    datasets = data_gouv.get_data()
    end = time.time()
    dump_to_json("data/datagouv.json", datasets)
    print(
        f"Fetched from DataGouv in {end - start}s with {data_gouv.request_count} requests"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch data from APIs")
    parser.add_argument(
        "-o",
        "--open-data-soft",
        help="Don't fetch from Open Data Soft",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "-s",
        "--data-sud",
        help="Don't fetch from Data Sud",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "-g",
        "--data-gouv",
        help="Don't fetch from Data Gouv",
        action="store_false",
        default=True,
    )
    args = parser.parse_args()

    if args.open_data_soft:
        print("Fetching from Open Data Soft")
        get_open_data_soft()
    if args.data_sud:
        print("Fetching from Data Sud")
        get_data_sud()
    if args.data_gouv:
        print("Fetching from Data Gouv")
        get_data_gouv()
