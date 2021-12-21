import asyncio
from urllib.request import urlopen
from urllib.error import URLError

from api import Api
from tools import remove_xml_tags


class DataGouv(Api):
    def __init__(self):
        super().__init__("https://www.data.gouv.fr/api/1")

        self.page_size = 100
        self.number_of_datasets = asyncio.run(self.get_number_of_datasets())

    async def get_number_of_datasets(self) -> int:
        """
        This method calls the API to fetch the total number of datasets on the website.

        :return: the total number of datasets hosted on data.gouv.fr
        """
        response = await self.fetch("/site/")

        return response["metrics"]["datasets"]

    @staticmethod
    async def get_columns(resources: list):
        columns = []
        for resource in resources:
            if resource["format"] != "csv":
                continue
            try:
                with urlopen(resource["url"]) as response:
                    columns = remove_xml_tags(
                        response.readline().decode().replace('"', "")
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
        return columns

    async def get_one_page(self, page_number: int) -> list:
        """
        This method returns all the datasets info featured on the specified page number.

        :param page_number: an integer corresponding to the page to be fetched
        :return: a list containing the datasets info featured on the query page number
        """
        page_data = await self.fetch(
            f"/datasets/?page_size={self.page_size}&page={page_number}"
        )
        return [
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
                    {"name": column_name, "type": var_type}
                    for column_name, var_type in await self.get_columns(
                        dataset["resources"]
                    )
                ],
            }
            for dataset in page_data["data"]
        ]

    async def __get_data(self) -> dict:
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
        This method gets the data on data.gouv.fr

        :return: a dictionary containing the datasets info hosted on data.gouv.fr
        """
        loop = asyncio.new_event_loop()
        coroutine = self.__get_data()
        return loop.run_until_complete(coroutine)
