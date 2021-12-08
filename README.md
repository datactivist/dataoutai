# dataoutai

Dataoutai : Représentations et mesures de similarité pour explorer et améliorer les catalogues de données ouvertes

## Mise en place

Créer un environnement conda contenant les bibliothèques utiles, avec les mêmes versions que sur Colab pour faciliter la portabilité du travail en local ou sur Colab :

```sh
conda env create -f environment.yml
```

À chaque session de travail, il faut activer l'environnement conda :

```sh
conda activate dataoutai
```

En fin de session :

```sh
conda deactivate
```

## Récupération des données

### DataSud

```python
from api import DataSud, dump_to_json

data_sud = DataSud()
datasets = data_sud.get_data()
dump_to_json("data/datasud.json", datasets)
```
