import sys
from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    """loads a single yml file

    Args:
        path (str): path to yml file

    Returns:
        Dict: dict with configurations (if fail, params is None)
    """
    try:
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)
    except:
        params = None
    return params
