import os.path
import pathlib


def mkdirs(filename: str) -> None:
    """
    Creates a new directory to store a given file.

    This function operates similarly to Java's File.mkdirs with the execution that it raises no alarms if the
    directory or any of its parents already exist.

    Args:
        filename:
            A string representing the relative or absolute system path to the file.
    """
    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
