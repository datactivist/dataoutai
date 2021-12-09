import asyncio
from typing import List
from urllib.error import HTTPError, URLError

from api import Api
from tools import remove_xml_tags


class DataSud(Api):
    def __init__(self):
        super().__init__("https://trouver.datasud.fr/api/3/action/")

    async def get_dataset_list(self) -> List[str]:
        """
        Fetch the list of all datasets available on the DataSud API
        :return: The list of all datasets
        """
        response = await self.fetch("/package_list")
        if response["success"]:
            return response["result"]
        else:
            return []

    async def get_columns(self, resources):
        column_names = set()
        columns = []
        for resource in resources:
            if resource["format"] != "CSV":
                continue
            try:
                response = await self.fetch(
                    f"/datastore_search?resource_id={resource['id']}"
                )
                if not response["success"]:
                    continue
                for column in response["result"]["fields"]:
                    if column["id"] in column_names:
                        continue
                    column_names.add(column["id"])
                    columns.append(column)
            except HTTPError:
                continue
            except URLError:
                print(resource["id"])
        return columns

    async def get_dataset_details(self, dataset):
        """
        Get the important dataset details:
        - Name
        - Keywords
        - Description
        - Columns
        :param dataset: The name of the dataset to get
        :return: The dataset important details
        """
        try:
            details = await self.fetch(f"/package_show?id={dataset}")
        except HTTPError:
            return
        if not details["success"]:
            return
        details = details["result"]
        try:
            columns = await self.get_columns(details["resources"])
        except HTTPError:
            columns = []
        return {
            "dataset_name": details["name"],
            "metadata": {
                "keywords": [
                    tag["name"] for tag in details["tags"] if tag["state"] == "active"
                ],
                "description": remove_xml_tags(details["notes"]),
            },
            "columns": columns,
        }

    async def __get_data(self) -> dict:
        """
        Private async method to get all the datasets from the DataSud API
        :return: All datasets and their count
        """
        dataset_list = await self.get_dataset_list()
        results = [
            result
            for result in await asyncio.gather(
                *map(self.get_dataset_details, dataset_list)
            )
            if result is not None
        ]
        return {"count": len(results), "datasets": results}

    def get_data(self):
        loop = asyncio.get_event_loop()
        coroutine = self.__get_data()
        return loop.run_until_complete(coroutine)
