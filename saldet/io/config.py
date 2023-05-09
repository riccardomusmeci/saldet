from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    """loads a single yml file

    Args:
        path (str): path to yml file

    Returns:
        Dict: yml dict
    """
    with open(path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    return params
