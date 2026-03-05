from typing import Callable

class CopyDictDescriptor:
    """
    A descriptor for a dict which always makes a copy of each item in the dictionary when the dictionary is accessed.

    This makes sure that the user cannot accidentally change a public variable.
    """
    def __init__(self, name, class_value=None):
        self._class_value = class_value
        self._private_name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            if self._class_value is None:
                return self
            values = self._class_value
        else:
            values = getattr(instance, self._private_name)
        new_values = {key: value.copy() for key, value in values.items()}
        return new_values

    def __set__(self, instance, value):
        setattr(instance, self._private_name, value)


class CopyAttributes:
    """
    A class which makes all non-private non-callable attributes use the `CopyDictDescriptor`.

    It implements `__init_class__` to change all the attributes to use `CopyDictDescriptor`.

    A non-private attribute is any attribute that does not start with `_`
    """
    def __init_subclass__(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not isinstance(value, Callable):
                setattr(cls, key, CopyDictDescriptor(key, value))