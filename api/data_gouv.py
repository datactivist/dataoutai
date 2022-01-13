import asyncio
from http.client import InvalidURL
from urllib.error import URLError, HTTPError

from tools import remove_xml_tags
from .api import Api


class DataGouv(Api):
    __metaclass__ = Api

    def __init__(self):
        """
        Instantiate the Data Gouv API.
        """
        super().__init__("https://www.data.gouv.fr/api/1")

        self.page_size = 100
        self.number_of_datasets = asyncio.run(self.get_number_of_datasets())

    async def get_number_of_datasets(self) -> int:
        """
        This method calls the API to fetch the total number of datasets on the website.

        :return: the total number of datasets hosted on data.gouv.fr
        """
        response = await self.fetch_json("/site/")

        return response["metrics"]["datasets"]

    async def get_columns(self, resources: list):
        columns = []
        for resource in resources:
            if resource["format"] != "csv":
                continue
            url = resource["url"]
            if len(url.strip()) == 0:
                continue
            try:
                with await self.fetch_csv(url) as response:
                    columns = remove_xml_tags(
                        response.readline().decode("iso-8859-1").replace('"', "")
                    )
                    if ";" in columns:
                        columns = columns.split(";")
                    elif "," in columns:
                        columns = columns.split(",")
                    else:
                        continue
                for column in columns:
                    if column in columns:
                        continue
                    columns.append(column, "")
            except URLError:
                continue
            except InvalidURL:
                continue
            except ConnectionResetError:
                continue
            except TimeoutError:
                continue
        return columns

    async def get_one_page(self, page_number: int) -> list:
        """
        This method returns all the datasets info featured on the specified page number.

        :param page_number: an integer corresponding to the page to be fetched
        :return: a list containing the datasets info featured on the query page number
        """
        try:
            page_data = await self.fetch_json(
                f"/datasets/?page_size={self.page_size}&page={page_number}"
            )
        except HTTPError:
            return []
        datasets = []
        for dataset in page_data["data"]:
            datasets.append(
                {
                    "dataset_name": dataset["title"],
                    "maintainer": dataset["owner"]["first_name"]
                    + dataset["owner"]["last_name"]
                    if dataset["owner"] is not None
                    else None,
                    "author": dataset["organization"].get("name", None)
                    if dataset["organization"] is not None
                    else None,
                    "licence": dataset["license"],
                    "frequency": dataset["frequency"],
                    "geographic_hold": dataset["spatial"].get("granularity", None)
                    if dataset["spatial"] is not None
                    else None,
                    "metadata": {
                        "keywords": [tag for tag in dataset["tags"]],
                        "description": remove_xml_tags(dataset["description"]),
                        "groups": [],
                    },
                    "columns": [
                        {"name": column_name, "type": None}
                        for column_name in await self.get_columns(dataset["resources"])
                    ],
                }
            )
        return datasets

    async def build_dataset_list_details(self) -> dict:
        """
        This method gets all the datasets info available on data.gouv.fr asynchronously.

        :return: a dictionary containing all the datasets info
        """
        result = await asyncio.gather(
            *map(
                self.get_one_page,
                range(1, self.number_of_datasets // self.page_size - 1),
            )
        )
        result = [dataset for dataset_list in result for dataset in dataset_list]

        return {
            "count": len(result),
            "datasets": result,
        }

    def get_data(self) -> dict:
        """
        This method calls an async function and wait until it finishes constructing the required data.
        :return: The dataset list with all the details.
        """
        loop = asyncio.new_event_loop()
        coroutine = self.build_dataset_list_details()
        return loop.run_until_complete(coroutine)
