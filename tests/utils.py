import inspect
import pathlib
import json


def load_data(filename, data_name, parser) -> list:
    with pathlib.Path(inspect.stack()[1].filename) as f:
        parent = f.parent
    with open(parent / f"{filename}.json") as f:
        data = json.load(f)
    return [parser(el) for el in data[data_name]]
