import datetime

STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"


def now() -> str:
    """Return current datetime as string format as "%Y-%m-%d-%H-%M-%S"

    Returns:
        str: current datetime (format "%Y-%m-%d-%H-%M-%S")
    """
    STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(STRFTIME_FORMAT)
