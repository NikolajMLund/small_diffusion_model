
import requests
import io
import pandas as pd


def import_BIL21():
    """

    """
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

def import_FAM55N():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "FAM55N",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "OMRÅDE",   "values": ["000"]},
            #{"code": "HUSSTØR",   "values": ["*"]},
            {"code": "Tid",    "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    BIL51 = pd.read_csv(io.StringIO(response.text), sep=";")
    return BIL51

def import_BIL52():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "BIL52",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "ALDER1", "values": ["*"]},
            {"code": "DRIV",   "values": ["*"]},
            {"code": "Tid",    "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    BIL52 = pd.read_csv(io.StringIO(response.text), sep=";")
    return BIL52

def import_NAHL2():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "NAHL2",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "TRANSAKT",  "values": ["B1GQD"]},  # BNP
            {"code": "PRISENHED", "values": ["LAN"]},     # Kædede priser 2020
            {"code": "Tid",       "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    NAHL2 = pd.read_csv(io.StringIO(response.text), sep=";")
    return NAHL2


def import_BEFOLK1():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "BEFOLK1",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "KØN",        "values": ["TOT"]},
            {"code": "ALDER",      "values": ["TOT"]},
            {"code": "CIVILSTAND", "values": ["TOT"]},
            {"code": "Tid",        "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    BEFOLK1 = pd.read_csv(io.StringIO(response.text), sep=";")
    return BEFOLK1


def import_BIL8():
    data_url = "https://api.statbank.dk/v1/data"

    payload = {
        "table": "BIL8",
        "format": "csv",
        "lang": "da",
        "variables": [
            {"code": "BILTYPE", "values": ["4000101002"]},  # Personbiler i alt
            {"code": "ALDER1",  "values": ["IALT"]},
            {"code": "Tid",     "values": ["*"]}
        ]
    }

    response = requests.post(data_url, json=payload)

    BIL8 = pd.read_csv(io.StringIO(response.text), sep=";")
    return BIL8


if __name__ == "__main__":
    BIL21 = import_BIL21()
    BIL51 = import_BIL51()
    FAM55N = import_FAM55N()



