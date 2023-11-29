from __future__ import annotations

import numpy
import numpy as np


def dbmean(a: float | np.ndarray) -> float:
    """
    Compute the arithmetic mean of a series of power quantity ratios, expressed as decibel levels.

    Parameters
    ----------
    a : float or array_like
        The levels to average.

    Returns
    -------
    logmean : float
        The mean of the ratios encoded in ``a``.

    See Also
    --------
    dbratios : Convert a series of decibel levels to the corresponding power quantity ratios.
    decibels : Return an empty array with shape and type of input.

    Examples
    --------
    >>> dbmean(40)
    40
    >>> dbmean(numpy.array([40, 35]))
    38.18
    """
    return dbratios(a).mean()


def dbratios(a: float | np.ndarray) -> float | np.ndarray:
    """
    Convert a series of decibel levels to the corresponding power quantity ratios.

    Parameters
    ----------
    a : float or array_like
        The levels to convert.

    Returns
    -------
    logmean : float or array_like
        The ratios encoded in ``a``.

    See Also
    --------
    decibels : Return an empty array with shape and type of input.

    Examples
    --------
    >>> dbratios(-10)
    0.1
    >>> dbratios(numpy.array([-10, 10]))
    numpy.array([0.1, 10.])
    """
    return np.full_like(a, 10) ** (0.1 * a)


def decibels(a: float | np.ndarray) -> float | np.ndarray:
    """
    Convert a series of power quantity ratios to the corresponding decibel levels.

    Parameters
    ----------
    a : float or array_like
        The ratios to convert.

    Returns
    -------
    logmean : float or array_like
        The levels encoded in ``a``.

    See Also
    --------
    dbratios : Return an empty array with shape and type of input.

    Examples
    --------
    >>> decibels(0.1)
    -10.0
    >>> decibels(numpy.array([0.1, 10]))
    numpy.array([-10.  10.])
    """
    return 10 * np.log10(a)
