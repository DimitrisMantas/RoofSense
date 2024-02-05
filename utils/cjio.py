import json
import sys
import tempfile
from types import TracebackType
from typing import IO, Never, Optional, Type

import cjio.cityjson
import requests


# noinspection PyUnusedLocal
def to_jsonl(
    response: requests.Response, *args: Never, **kwargs: Never
) -> requests.Response:
    lines = response.json()
    with tempfile.TemporaryFile("w+") as f:
        f.write(f"{json.dumps(lines['metadata'])}\n")
        if "feature" in lines:
            f.write(f"{json.dumps(lines['feature'])} \n")
        if "features" in lines:
            for feat in lines["features"]:
                f.write(f"{json.dumps(feat)}\n")
        # NOTE - Reset the offset into the file to the beginning so that it will be
        #        read in its entirety when the standard input is pointed to it.
        f.seek(0)
        with _stdin(f):
            lines = cjio.cityjson.read_stdin()
    response._content = json.dumps(lines.j).encode()
    return response


class _stdin:
    def __init__(self, f: IO[str]) -> None:
        self._f = f

    def __enter__(self) -> None:
        sys.stdin = self._f

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        sys.stdin = sys.__stdin__
        # NOTE: An IOError instance is raised when the response does not correspond
        #       to the CityJSON Feature schema.
        return isinstance(exc_value, IOError)
