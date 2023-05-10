from saldet.io import load_config


def test_load_config():
    config_path = "tests/data/config/test.yaml"
    config = load_config(config_path)
    assert config["model"]["model_name"] == "u2net_lite", f"Error in loading config"


def test_exception_config():
    config_path = "test.yaml"
    config = load_config(config_path)
    assert config is None
