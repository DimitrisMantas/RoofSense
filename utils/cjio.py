import json
import sys
import tempfile
import typing

import cjio.cityjson
import requests


def to_jsonl(response: requests.Response, *args, **kwargs):
    j = response.json()
    with tempfile.TemporaryFile("w+") as f:
        f.write(f"{json.dumps(j['metadata'])}\n")
        if "feature" in j:
            f.write(f"{json.dumps(j['feature'])} \n")
        if "features" in j:
            for feat in j["features"]:
                f.write(f"{json.dumps(feat)}\n")

        # NOTE - Reset the offset into the file to the beginning so that it will be
        #        read in its entirety when the standard input is pointed to it.
        f.seek(0)
        with stdin(f):
            j = cjio.cityjson.read_stdin()

    response._content = json.dumps(j.j).encode()
    return response


class stdin:
    def __init__(self, file: typing.TextIO) -> None:
        self._file = file

    def __enter__(self) -> None:
        sys.stdin = self._file

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdin = sys.__stdin__
