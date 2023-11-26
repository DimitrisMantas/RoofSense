import concurrent.futures
import logging
import pathlib
import sys
import typing

import requests
import tqdm


# TODO - Reformat, finalize function and variable names, and add documentation.

class _FileDownloader:
    # FIXME - Add timeout and retry options.
    def __init__(self, overwrite: bool = False) -> None:
        self._overwrite = overwrite

    def download(self) -> None:
        raise NotImplementedError

    def _obtain(self, address: str, filename: str) -> None:
        # FIXME - Refactor this guard clause into a separate function.
        if is_file(filename) and not self._overwrite:
            return

        with requests.get(address, stream=True) as response:
            _FileDownloader._handle(response)

            # NOTE - Write the response inside a with-block so that the request is not consumed prematurely,
            #        resulting in the connection to the server breaking.
            _FileDownloader._write(response, filename)

    @staticmethod
    def _handle(response: requests.Response) -> None:
        # FIXME - Hande the possible response status codes.
        #        https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#information_responses
        if 100 >= response.status_code >= 199:
            pass

        if 300 >= response.status_code >= 399:
            pass

        if 300 >= response.status_code >= 399:
            pass

        if 300 >= response.status_code >= 399:
            pass

        if 300 >= response.status_code >= 399:
            pass

    # FIXME - Write the chunks into a temporary file.
    @staticmethod
    def _write(response: requests.Response, filename: str) -> None:
        try:
            with pathlib.Path(filename).open("wb") as f:
                for chunk in response.iter_content(chunk_size=None):
                    f.write(chunk)
        except (FileNotFoundError, IsADirectoryError) as e:
            logging.error(e, exc_info=True)


class BlockingFileDownloader(_FileDownloader):
    def __init__(self, address: str, filename: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.address = address
        self.filename = filename

    def download(self) -> None:
        if is_file(self.filename) and not self._overwrite:
            return

        with requests.get(self.address, stream=True) as response:
            self._handle(response)

            # NOTE - Write the response inside a with-block so that the request is not consumed prematurely,
            #        resulting in the connection to the server breaking.
            self._write(response, self.filename)

    # FIXME - Write the chunks into a temporary file.
    # FIXME - Refactor this function into separate ones.
    @staticmethod
    def _write(response: requests.Response, filename: str) -> None:
        total = int(response.headers["content-length"]) if response.headers["content-length"] is not None else None
        with tqdm.tqdm(desc="Download Progress", total=total, unit="iB", unit_scale=True,
                       unit_divisor=1024) as progress:
            try:
                with pathlib.Path(filename).open("wb") as f:
                    for chunk in response.iter_content(chunk_size=None):
                        f.write(chunk)
                        progress.update(len(chunk))
            except (FileNotFoundError, IsADirectoryError) as e:
                logging.error(e, exc_info=True)


class ThreadedFileDownloader(_FileDownloader):
    def __init__(self, addresses: typing.Collection[str], filenames: typing.Collection[str],

                 max_conns: int = None,

                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._addresses = addresses
        self._filenames = filenames

        self._max_conns = None if max_conns is None else max_conns

    def download(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_conns) as executor:
            tasks = {executor.submit(self._obtain, address, filename): (address, filename) for address, filename in
                     zip(self._addresses, self._filenames)}
            # FIXME - Color the entire progress bar white.
            with tqdm.tqdm(desc="Download Progress", total=len(tasks), unit="File") as progress:
                for task in concurrent.futures.as_completed(tasks):
                    address, filename = tasks[task]
                    try:
                        task.result()
                    # FIXME - Find out when this error is raised.
                    except concurrent.futures.CancelledError:
                        pass

                    progress.update(1)


class stdin:
    def __init__(self, f: typing.TextIO) -> None:
        self._f = f

    def __enter__(self) -> None:
        sys.stdin = self._f

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdin = sys.__stdin__


def is_file(filename: str) -> bool:
    return pathlib.Path(filename).is_file()


def mkdirs(filename: str) -> None:
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
