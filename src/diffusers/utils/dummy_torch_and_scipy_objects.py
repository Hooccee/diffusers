# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import DummyObject, requires_backends


class LMSDiscreteScheduler(metaclass=DummyObject):
    _backends = ["torch", "scipy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "scipy"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "scipy"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "scipy"])
