"""
Retry and exponential-backoff utilities for transient network/API errors.

Usage
-----
    from src.data.retry import with_retry

    @with_retry(max_retries=3, base_delay=1.0)
    def my_fetch_function(...):
        ...

Or call the helper directly:

    result = retry_call(my_fetch_function, args, kwargs, max_retries=3)
"""

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, TypeVar

import structlog

from src.data.provider import DataFetchError

log = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Default constants ──────────────────────────────────────────────────────
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BASE_DELAY: float = 1.0   # seconds
DEFAULT_MAX_DELAY: float = 30.0   # seconds
DEFAULT_BACKOFF_FACTOR: float = 2.0


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True if *exc* is a transient error that is safe to retry.

    Rules:
    - DataFetchError with a 4xx status code → NOT retryable.
    - DataFetchError with 5xx or no status  → retryable.
    - ConnectionError, TimeoutError, OSError → retryable.
    - Any other exception                   → NOT retried (fail fast).
    """
    if isinstance(exc, DataFetchError):
        return exc.is_retryable()
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _jittered_delay(attempt: int, base: float, factor: float, cap: float) -> float:
    """
    Compute the sleep duration for *attempt* (0-indexed) using full-jitter
    exponential backoff: uniform in [0, min(cap, base * factor**attempt)].
    """
    ceiling = min(cap, base * (factor ** attempt))
    return random.uniform(0, ceiling)


def retry_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> Any:
    """
    Call *fn* with *args* / *kwargs*, retrying on transient errors.

    Parameters
    ----------
    fn : callable
        The function to call.
    args : tuple
        Positional arguments.
    kwargs : dict | None
        Keyword arguments.
    max_retries : int
        Maximum number of retry attempts after the initial call fails.
        Total attempts = max_retries + 1.
    base_delay : float
        Base sleep duration in seconds.
    max_delay : float
        Maximum sleep cap in seconds.
    backoff_factor : float
        Multiplicative backoff factor.

    Returns
    -------
    Any
        Return value of *fn*.

    Raises
    ------
    DataFetchError
        Re-raised after all retries are exhausted or on non-retryable errors.
    Exception
        Any non-retryable exception is propagated immediately.
    """
    if kwargs is None:
        kwargs = {}

    last_exc: BaseException | None = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                log.debug(
                    "retry_call.non_retryable",
                    fn=getattr(fn, "__qualname__", str(fn)),
                    attempt=attempt,
                    exc_type=type(exc).__name__,
                )
                raise

            remaining = total_attempts - attempt - 1
            if remaining == 0:
                log.warning(
                    "retry_call.exhausted",
                    fn=getattr(fn, "__qualname__", str(fn)),
                    attempts=total_attempts,
                    exc=str(exc),
                )
                raise

            delay = _jittered_delay(attempt, base_delay, backoff_factor, max_delay)
            log.info(
                "retry_call.retrying",
                fn=getattr(fn, "__qualname__", str(fn)),
                attempt=attempt + 1,
                remaining=remaining,
                delay_s=round(delay, 3),
                exc=str(exc),
            )
            time.sleep(delay)

    # Should never reach here, but satisfy the type-checker.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_call: unexpected code path")  # pragma: no cover


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> Callable[[F], F]:
    """
    Decorator factory that wraps a function with retry/backoff logic.

    Example
    -------
        @with_retry(max_retries=3, base_delay=0.5)
        def fetch_data(ticker: str) -> OHLCVSeries:
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_call(
                fn,
                args=args,
                kwargs=kwargs,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            )

        return wrapper  # type: ignore[return-value]

    return decorator
