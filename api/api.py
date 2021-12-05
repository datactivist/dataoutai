import json
import urllib.request


class Api:
    def __init__(self, base_url: str):
        if base_url.endswith('/'):
            self.base_url = base_url[:-1]
        else:
            self.base_url = base_url

    def fetch(self, path: str):
        with urllib.request.urlopen(f"{self.base_url}{path}") as url:
            data = json.loads(url.read().decode())
        print(data)
