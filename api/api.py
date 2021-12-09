import concurrent.futures
import json
import urllib.request
from abc import abstractmethod

from tools import run_in_executor


class Api:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=120)

    def __init__(self, base_url: str):
        """
        Instantiate an API to fetch resources
        :param base_url: The API base URL
        """
        if base_url.endswith("/"):
            self.base_url = base_url[:-1]
        else:
            self.base_url = base_url

    @run_in_executor(executor)
    def __get_from_url(self, path: str):
        """
        Calls an API endpoint with a GET request
        :param path: Path following the API base URL
        :return: The JSON payload
        """
        with urllib.request.urlopen(f"{self.base_url}{path}") as url:
            data = json.loads(url.read().decode(encoding="utf8"))
        return data

    async def fetch(self, path: str) -> dict:
        return await self.__get_from_url(path)

    @abstractmethod
    def get_data(self):
        pass


def dump_to_json(
    file_path: str,
    data: dict,
    encoding: str = "utf8",
    indent: int = 2,
    ensure_ascii: bool = False,
):
    """
    Dumps a Python dict to a JSON file
    :param file_path: The output file path
    :param data: The data to dump
    :param encoding: Encoding to use when writing the file
    :param indent: Ident size for the JSON
    :param ensure_ascii: See the json library documentation for more info
    """
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, fp=f, indent=indent, ensure_ascii=ensure_ascii)
