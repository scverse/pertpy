from collections.abc import Callable
from functools import singledispatch, update_wrapper
from typing import Any, Protocol, TypeVar, runtime_checkable

R = TypeVar("R")


@runtime_checkable
class SingleDispatchFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...

    def register(self, cls: type, func: Callable[..., R] = None) -> Callable[..., R]: ...


def methdispatch(func: Callable[..., R]) -> SingleDispatchFunction[R]:
    """Wrapper of singledispatch that works on instance methods."""
    dispatcher = singledispatch(func)

    def wrapper(*args: Any, **kwargs: Any) -> R:
        return dispatcher.dispatch(args[1].__class__)(*args, **kwargs)

    wrapper.register = dispatcher.register  # type: ignore
    update_wrapper(wrapper, func)

    return wrapper  # type: ignore
