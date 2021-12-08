import json
import urllib.request
from abc import abstractmethod


class Api:
    def __init__(self, base_url: str):
        if base_url.endswith("/"):
            self.base_url = base_url[:-1]
        else:
            self.base_url = base_url

    async def fetch(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}") as url:
            data = json.loads(url.read().decode(encoding="utf8"))
        return data

    @abstractmethod
    def get_data(self):
        pass


def dump_to_json(file_path: str, data: dict):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, fp=f)
