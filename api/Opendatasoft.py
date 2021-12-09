import asyncio
from typing import List
from tools import remove_xml_tags
from api import Api


class Opendatasoft(Api):
    def __init__(self):
        super().__init__("https://public.opendatasoft.com/api/v2/catalog/")

    async def get_dataset_list(self, nb_per_query: int = 50) -> List[str]:
        """
        Fetch the list of all url queries of the datasets available on the Opendatasoft API
        :return: The list of all queries url
        :nb_per_query The number of dataset per query (max 100)
        """
        if nb_per_query > 100:
            nb_per_query = 100
        response = await self.fetch("/datasets?limit=1&offset=0")
        count = response["total_count"]
        return [
            f"/datasets?limit={nb_per_query}&offset={offset}"
            for offset in range(0, count, nb_per_query)
        ]

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
        datasets = await self.fetch(dataset_url)
        clean_data = []
        for dataset in datasets["datasets"]:
            clean_data.append(
                {
                    "dataset_name": f'{dataset["dataset"]["dataset_id"]}',
                    "metadata": {
                        "keywords": [
                            []
                            if dataset["dataset"]["metas"]["default"]["keyword"] is None
                            else [
                                keyword
                                for keyword in dataset["dataset"]["metas"]["default"][
                                    "keyword"
                                ]
                            ]
                        ],
                        "description": ""
                        if dataset["dataset"]["metas"]["default"]["description"] is None
                        else f'{remove_xml_tags(dataset["dataset"]["metas"]["default"]["description"])}',
                    },
                    "columns": [
                        {"name": annotation["name"], "type": annotation["type"]}
                        for annotation in dataset["dataset"]["fields"]
                    ],
                }
            )
        return clean_data

    async def __get_data(self) -> dict:
        """
        Private async method to get all the datasets from the DataSud API
        :return: All datasets and their count
        """
        dataset_list = self.get_dataset_list()
        results = await asyncio.gather(
            *map(self.get_dataset_details, await dataset_list)
        )
        response = await self.fetch("/datasets?limit=1&offset=0")
        count = response["total_count"]
        return {
            "count": count,
            "datasets": [item for sublist in results for item in sublist],
        }

    def get_data(self) -> dict:
        loop = asyncio.get_event_loop()
        coroutine = self.__get_data()
        return loop.run_until_complete(coroutine)
