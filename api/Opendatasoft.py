import asyncio
from typing import List
from urllib.error import HTTPError

from tools import remove_xml_tags
from .api import Api


class Opendatasoft(Api):
    __metaclass__ = Api

    def __init__(self):
        """
        Instantiate the Open Data Soft API.
        """
        super().__init__("https://data.opendatasoft.com/api/v2/catalog/")

    async def get_dataset_list(self, nb_per_query: int = 50) -> List[str]:
        """
        Fetch the list of all url queries of the datasets available on the Opendatasoft API
        :return: The list of all queries url
        :nb_per_query The number of dataset per query (max 100)
        """
        if nb_per_query > 100:
            nb_per_query = 100
        response = await self.fetch_json("/datasets?limit=1&offset=0")
        count = response["total_count"]
        return [
            f"/datasets?limit={nb_per_query}&offset={offset}"
            for offset in range(0, count, nb_per_query)
        ]

    @staticmethod
    def __check_frequency(frequency: str):
        if frequency:
            if frequency.startswith("http"):
                return frequency.split("/")[-1]
            else:
                return frequency
        else:
            return None

    async def get_dataset_details(self, dataset_url: str) -> List[dict]:
        """
        Get the important dataset details:
        - Name
        - Keywords
        - Description
        - Columns
        :param dataset_url: The url query of the dataset to get
        :return: The dataset important details
        """
        try:
            datasets = await self.fetch_json(dataset_url)
            clean_data = []
            for dataset in datasets["datasets"]:
                clean_data.append(
                    {
                        "dataset_name": f'{dataset["dataset"]["metas"]["default"]["title"]}',
                        "maintainer": None
                        if "custom" not in dataset["dataset"]["metas"]["default"]
                        else dataset["dataset"]["metas"]["default"]["custom"][
                            "gestionnaire"
                        ],
                        "author": None
                        if "publisher" not in dataset["dataset"]["metas"]["default"]
                        else dataset["dataset"]["metas"]["default"]["publisher"],
                        "licence": None
                        if "licence" not in dataset["dataset"]["metas"]["default"]
                        else dataset["dataset"]["metas"]["default"]["licence"],
                        "frequency": None
                        if "accrualperiodicity"
                        not in dataset["dataset"]["metas"]["dcat"]
                        else self.__check_frequency(
                            dataset["dataset"]["metas"]["dcat"]["accrualperiodicity"]
                        ),
                        "geographic_hold": None
                        if "territory" not in dataset["dataset"]["metas"]["default"]
                        else dataset["dataset"]["metas"]["default"]["territory"],
                        "metadata": {
                            "keywords": None
                            if dataset["dataset"]["metas"]["default"]["keyword"] is None
                            else dataset["dataset"]["metas"]["default"]["keyword"],
                            "description": None
                            if dataset["dataset"]["metas"]["default"]["description"]
                            is None
                            else f'{remove_xml_tags(dataset["dataset"]["metas"]["default"]["description"])}',
                            "groups": [],
                        },
                        "columns": [
                            {"name": annotation["name"], "type": annotation["type"]}
                            for annotation in dataset["dataset"]["fields"]
                        ],
                    }
                )
            return clean_data
        except HTTPError:
            return []

    async def build_dataset_list_details(self) -> dict:
        """
        Private async method to get all the datasets from the DataSud API
        :return: All datasets and their count
        """
        dataset_list = self.get_dataset_list(99)
        results = await asyncio.gather(
            *map(self.get_dataset_details, await dataset_list)
        )
        response = await self.fetch_json("/datasets?limit=1&offset=0")
        count = response["total_count"]
        return {
            "count": count,
            "datasets": [item for sublist in results for item in sublist],
        }
