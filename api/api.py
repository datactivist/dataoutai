import json
import urllib.request
from abc import abstractmethod


class Api:
    def __init__(self, base_url: str):
        """
        Instantiate an API to fetch resources
        :param base_url: The API base URL
        """
        if base_url.endswith("/"):
            self.base_url = base_url[:-1]
        else:
            self.base_url = base_url

    async def fetch(self, path: str) -> dict:
        """
        Calls an API endpoint with a GET request
        :param path: Path following the API base URL
        :return: The JSON payload
        """
        with urllib.request.urlopen(f"{self.base_url}{path}") as url:
            data = json.loads(url.read().decode(encoding="utf8"))
        return data

    @abstractmethod
    def get_data(self):
        pass


def dump_to_json(file_path: str, data: dict):
    """
    Dumps a Python dict to a JSON file
    :param file_path: The output file path
    :param data: The data to dump
    """
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, fp=f)
