import asyncio
import concurrent.futures
import ssl
from abc import abstractmethod
from json import loads
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit, quote
from urllib.request import urlopen

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
        self.request_count = 0

    @run_in_executor(executor)
    def __get_csv(self, url: str):
        def clean_url():
            cleaned_url = urlsplit(url)
            cleaned_url = list(cleaned_url)
            url_parts = cleaned_url[:2]
            for part in cleaned_url[2:]:
                url_parts.append(quote(part))
            return urlunsplit(url_parts)

        self.request_count += 1

        try:
            return urlopen(url)
        except UnicodeEncodeError:
            url = clean_url()
            return urlopen(url)

    async def fetch_csv(self, url: str) -> dict:
        return await self.__get_csv(url)

    @run_in_executor(executor)
    def __get_json(self, path: str):
        """
        Calls an API endpoint with a GET request
        :param path: Path following the API base URL
        :return: The JSON payload
        """
        self.request_count += 1
        try:
            with urlopen(f"{self.base_url}{path}") as url:
                data = loads(url.read().decode(encoding="utf8"))
        except URLError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(f"{self.base_url}{path}", context=ctx) as url:
                data = loads(url.read().decode(encoding="utf8"))
        return data

    async def fetch_json(self, path: str) -> dict:
        return await self.__get_json(path)

    @abstractmethod
    async def build_dataset_list_details(self) -> dict:
        """This method build the dataset list with all the details required"""

    def get_data(self) -> dict:
        """
        This method calls an async function and wait until it finishes constructing the required data.
        :return: The dataset list with all the details.
        """
        loop = asyncio.get_event_loop()
        coroutine = self.build_dataset_list_details()
        return loop.run_until_complete(coroutine)
