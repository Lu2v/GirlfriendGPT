
from .luna import luna
from .sacha import sacha
from .angele import angele
from .aura import aura

__all__ = [
    "sacha",
    "luna",
    "angele",
    "aura",
    "get_personality",
]


def get_personality(name: str):
    try:
        return {
            "luna": luna,
            "sacha": sacha,
            "angele": angele,
            "Angèle": angele,
            "aura": aura,
        }[name]
    except KeyError:
        raise Exception(
            f"La personalidad '{name}' no existe."
        ) from None
