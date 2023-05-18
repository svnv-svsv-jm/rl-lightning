__all__ = ["wrong_type_err", "wrong_type_err_msg"]

import typing as ty


def wrong_type_err(target: ty.Any, target_type: ty.Any, extra: str = "") -> None:
    """Raises an error if the given object is not of the target type."""
    assert isinstance(target, target_type), wrong_type_err_msg(
        target, target_type, extra
    )


def wrong_type_err_msg(target: ty.Any, correct_type: ty.Any, extra: str = None) -> str:
    """Custom error message for wrong type."""
    if extra is None:
        extra = ""
    msg = f"Variable should be of type {correct_type} but was found of type {type(target)}. {extra}"
    return msg
