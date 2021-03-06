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
from api import DataSud
from tools import dump_to_json

data_sud = DataSud()
datasets = data_sud.get_data()
dump_to_json("data/datasud.json", datasets)
```

### Data.Gouv

```python
from api import DataGouv
from tools import dump_to_json

data_gouv = DataGouv()
datasets = data_gouv.get_data()
dump_to_json("data/datagouv.json", datasets)
```

### Opendatasoft

```python
from api import Opendatasoft
from tools import dump_to_json

open_data_soft = Opendatasoft()
datasets = open_data_soft.get_data()
dump_to_json("data/opendatasoft.json", datasets)
```