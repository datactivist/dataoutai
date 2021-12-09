from api import dump_to_json
from api import Api

import asyncio


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
                "metadata": {
                    "keywords": [tag for tag in dataset["tags"]],
                    "description": dataset["description"],
                },
                "columns": [],
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
                range(1, self.number_of_datasets // self.page_size),
            )
        )

        return {"count": self.number_of_datasets, "datasets": result}

    def get_data(self) -> dict:
        """
        This method gets the data on data.gouv.fr

        :return: a dictionary containing the datasets info hosted on data.gouv.fr
        """
        loop = asyncio.new_event_loop()
        coroutine = self.__get_data()
        return loop.run_until_complete(coroutine)
