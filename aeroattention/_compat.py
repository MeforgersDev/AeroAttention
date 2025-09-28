from __future__ import annotations

"""Utilities for working with the optional NumPy dependency."""

from typing import Any, TYPE_CHECKING

try:  # pragma: no cover - exercised when numpy is available
    import numpy as _np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when numpy is missing
    _np = None

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as np


np = _np


def has_numpy() -> bool:
    """Return ``True`` when NumPy is importable."""

    return np is not None


def require_numpy(feature: str) -> "np":
    """Return the imported NumPy module or raise a helpful error.

    Parameters
    ----------
    feature:
        Description of the feature that requires NumPy. Used to provide a
        descriptive error message when the dependency is missing.
    """

    if np is None:
        raise ModuleNotFoundError(
            f"NumPy is required for {feature}. Please install the 'numpy' package."
        )
    return np  # type: ignore[return-value]


def is_numpy_array(value: Any) -> bool:
    """Check whether *value* is an instance of ``numpy.ndarray``."""

    return np is not None and isinstance(value, np.ndarray)