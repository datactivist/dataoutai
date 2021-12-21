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
                    if column in columns:
                        continue
                    columns.append(column)
            except HTTPError:
                continue
            except URLError:
                continue
        return columns

    async def get_dataset_details(self, dataset):
        """
        Get the important dataset details:
        - Name
        - Maintainer
        - Author
        - Licence
        - Update frequency
        - Geographical granularity
        - Metadata
            - Keywords
            - Description
            - Groups
        - Columns
        :param dataset: The name of the dataset to get
        :return: The dataset important details
        """
        try:
            details = await self.fetch(f"/package_show?id={dataset}")
        except HTTPError:
            return

        def extract_field(column, default=None):
            return details.get(column, default)

        if not extract_field("success"):
            return
        details = extract_field("result")

        columns = await self.get_columns(details["resources"])

        return {
            "dataset_name": details["title"],
            "maintainer": extract_field("maintainer"),
            "author": details["author"],
            "licence": extract_field("licence"),
            "frequency": extract_field("frequency"),
            "geographic_hold": extract_field("granularity"),
            "metadata": {
                "keywords": [
                    tag["name"] for tag in details["tags"] if tag["state"] == "active"
                ],
                "description": remove_xml_tags(details["notes"]),
                "groups": [group["title"] for group in details["groups"]],
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
