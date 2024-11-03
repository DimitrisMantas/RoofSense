from __future__ import annotations

import abc
import concurrent.futures
import logging
import os
import pathlib
import typing
import warnings
from os import PathLike
from typing import Union

import requests
import requests.adapters
import tqdm
import urllib3

# TODO - Reformat, finalize function and variable names, and add documentation.

_Timeout = Union[float, tuple[float, float], tuple[float, None]]
_HookCallback = typing.Callable[[requests.Response], typing.Any]


def get_default_data_dir(version: str) -> str:
    path = (
            pathlib.Path.home() / ".roofsense" / version
    ).as_posix()
    os.makedirs(
        path, exist_ok=True
    )
    return path


def confirm_write_op(
        path: str, type: typing.Literal["dir", "file"], overwrite: bool = False
) -> bool:
    """Determine whether an upcoming write-to-disk operation should be performed.

    If the path to the destination location exists in the system and points to a
    resource of the provided type, this function returns the value of the
    ``overwrite`` flag. Otherwise, if the specified path is of an incorrect ype,
    a relevant exception is raised. Finally, if the output location does not exist,
    the value of ``overwrite`` is ignored and ``True`` is always returned. In this
    case, if ``overwrite == False`` a relevant warning is issued. Required
    directories are created automatically.

    Args:
        path:
            The path to the destination location.
        type:
            The resource type at the destination location.
        overwrite:
            Whether the resource at the destination location should be overwritten
            assuming that it already exists.

    Returns:
        ``True`` if the operation should be performed; ``False`` otherwise.
    """
    predicate = os.path.isdir if type == "dir" else os.path.isfile

    if predicate(path):
        return overwrite

    if os.path.exists(path):
        msg = (
            f"The specified path: {path!r} exists in the system but does not point to "
            f"a valid {dict(dir='directory', file='file')[type]}."
        )
        raise ValueError(msg)

    if not overwrite:
        msg = (
            f"The specified path: {path!r} does not exist in the system. The value of "
            f"the 'overwrite' flag will be ignored."
        )
        warnings.warn(msg, UserWarning)

    if type == "dir":
        os.makedirs(path, exist_ok=True)

    return True


class FileDownloader(abc.ABC):
    def __init__(
            self,
            session: requests.Session,
            overwrite: bool = False,
            timeout: _Timeout | None = (3.05, None),
            callbacks: _HookCallback | typing.Collection[_HookCallback] | None = None,
            # FIXME - Find a way to turn progress reporting on and off automatically instead of leaving it up to
            #         the user to decide.
            report_progress: bool = True,
    ) -> None:
        self._session = session
        # FIXME - Expose these options to the user.
        self._retries = urllib3.util.Retry(
            total=3, backoff_factor=0.5, backoff_jitter=0.5
        )
        self._session.mount(
            "https://", requests.adapters.HTTPAdapter(max_retries=self._retries)
        )

        self._overwrite = overwrite
        self._timeout = timeout
        self._callbacks = {"response": callbacks}

        self._preport = report_progress

    @abc.abstractmethod
    def download(self) -> None:
        pass

    def _fetch(self, url: str, filename: str) -> None:
        if exists(filename) and not self._overwrite:
            return

        with self._session.get(
                url, timeout=self._timeout, hooks=self._callbacks, stream=True, verify=False
        ) as response:
            self._handle(response)

            # NOTE - Write the response inside a with-block so that the request is not consumed prematurely resulting
            #        in the connection to the server breaking.
            try:
                with pathlib.Path(filename).open("wb") as f:
                    self._write(response, f)
            except (FileNotFoundError, IsADirectoryError) as e:
                logging.error(e, exc_info=True)

    # FIXME - Hande the possible response status codes.
    #        https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#information_responses
    @staticmethod
    def _handle(response: requests.Response) -> None:
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
    # FIXME - Refactor this function into separate ones.
    def _write(self, response: requests.Response, file: typing.BinaryIO) -> None:
        content_length = response.headers["content-length"]
        # FIXME - Color the entire progress bar white.
        with tqdm.tqdm(
                desc="Download Progress",
                total=int(content_length) if content_length is not None else None,
                disable=not self._preport,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
        ) as progress_bar:
            # NOTE - Write the response as it arrives instead of splitting it into possibly smaller-than-received chunks
            #        resulting in additional I/O operations.
            for chunk in response.iter_content(chunk_size=None):
                file.write(chunk)
                progress_bar.update(len(chunk))


class BlockingFileDownloader(FileDownloader):
    def __init__(self, url: str, filename: str | bytes | PathLike, **kwargs) -> None:
        super().__init__(**kwargs)

        self._url = url
        self._filename = filename

    def download(self) -> None:
        self._fetch(self._url, self._filename)


class ThreadedFileDownloader(FileDownloader):
    def __init__(
            self,
            urls: typing.Collection[str],
            filenames: typing.Collection[str | bytes | PathLike],
            max_conns: int | None = None,
            **kwargs,
    ) -> None:
        super().__init__(report_progress=False, **kwargs)

        self._urls = urls
        self._filenames = filenames

        self._max_conns = max_conns

    def download(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_conns
        ) as executor:
            futures = {
                executor.submit(self._fetch, url, filename): (url, filename)
                for url, filename in zip(self._urls, self._filenames)
            }
            # FIXME - Color the entire progress bar white.
            with tqdm.tqdm(
                    desc="Download Progress", total=len(futures), unit="File"
            ) as progress_bar:
                for task in concurrent.futures.as_completed(futures):
                    address, filename = futures[task]
                    try:
                        task.result()
                    # FIXME - Find when this error is raised.
                    except concurrent.futures.CancelledError:
                        pass
                    progress_bar.update()


def exists(filename: str) -> bool:
    return pathlib.Path(filename).is_file()


def mkdirs(filename: str) -> None:
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
