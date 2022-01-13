import json
import random
from typing import Dict, Generator, List


def __flatten_dict_gen(d: Dict) -> Generator:
    for k, v in d.items():
        new_key = k
        if isinstance(v, Dict):
            yield from __flatten_dict(v).items()
        else:
            yield new_key, v


def __flatten_dict(d: Dict) -> Dict:
    return dict(__flatten_dict_gen(d))


def filter_data(
    file_path: str, tags_filters: List[str], random_data: int = None
) -> Dict[str, str]:
    """
    Filter the datas to the original dataset to string for the tfidf vectorizer
    :parameter file_path: The path of the .json datasets file
    :parameter tags_filters: List of tags to include in the transformed datas
    :param random_data: Number of random data picked from the transformed datas
    :return Dict of {"id_data" : filtered_data}
    """

    with open(file_path, encoding="utf8") as f:
        datas = json.load(f)

    datasets = datas["datasets"]
    tags = {tag: "" for tag in tags_filters}
    transformed_data = {}

    for d in datasets:
        flatten_d = __flatten_dict(d)

        for tag in tags:
            if type(flatten_d[tag]) is list:
                tags[tag] = " ".join(flatten_d[tag])
            else:
                tags[tag] = flatten_d[tag]

        transformed_data[flatten_d["dataset_name"]] = " ".join(
            filter(None, tags.values())
        )

    if random_data is not None:
        random.seed(42)
        dict_key = random.sample(list(transformed_data), random_data)
        transformed_data = {k: transformed_data[k] for k in dict_key}

    return transformed_data
