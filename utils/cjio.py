#          Copyright © 2023 Dimitris Mantas
#
#          This file is part of RoofSense.
#
#          This program is free software: you can redistribute it and/or modify
#          it under the terms of the GNU General Public License as published by
#          the Free Software Foundation, either version 3 of the License, or
#          (at your option) any later version.
#
#          This program is distributed in the hope that it will be useful,
#          but WITHOUT ANY WARRANTY; without even the implied warranty of
#          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#          GNU General Public License for more details.
#
#          You should have received a copy of the GNU General Public License
#          along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

        # NOTE - Reset the offset into the file to the beginning so that it will be read in its entirety when the
        #        standard input is pointed to it.
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
