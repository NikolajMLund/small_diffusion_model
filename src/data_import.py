
import requests
import io
import pandas as pd


def import_BIL21():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "BIL21",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "ALDER1", "values": ["*"]},
            {"code": "DRIV",   "values": ["*"]},
            {"code": "Tid",    "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    BIL21 = pd.read_csv(io.StringIO(response.text), sep=";")
    return BIL21


def import_BIL51():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "BIL51",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "EJER",   "values": ["*"]},
            {"code": "DRIV",   "values": ["*"]},
            {"code": "Tid",    "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    BIL51 = pd.read_csv(io.StringIO(response.text), sep=";")
    return BIL51

if __name__ == "__main__":
    BIL21 = import_BIL21()
    BIL51 = import_BIL51()
    breakpoint()


