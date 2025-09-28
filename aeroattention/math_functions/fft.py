from __future__ import annotations

from .._compat import require_numpy


def fft(x):
    """Compute the discrete Fourier transform of ``x`` using NumPy."""

    numpy = require_numpy("fft")
    x = numpy.asarray(x, dtype=complex)
    n = x.shape[0]

    if n <= 1:
        return x

    x_even = fft(x[::2])
    x_odd = fft(x[1::2])
    factor = numpy.exp(-2j * numpy.pi * numpy.arange(n) / n)
    return numpy.concatenate((x_even + factor[: n // 2] * x_odd, x_even + factor[n // 2 :] * x_odd))


def ifft(x):
    """Compute the inverse discrete Fourier transform of ``x`` using NumPy."""

    numpy = require_numpy("ifft")
    x_conj = numpy.conjugate(x)
    result = fft(x_conj)
    result = numpy.conjugate(result)
    result = result / x.shape[0]
    return result
