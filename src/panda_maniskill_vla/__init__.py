"""Local ManiSkill environment tools for Panda control experiments."""

__all__ = ["PandaEnvConfig"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy import so clients (e.g. the VLA client) can use the protobuf messages
    # without having ManiSkill installed in their environment.
    if name == "PandaEnvConfig":
        from panda_maniskill_vla.panda_env import PandaEnvConfig

        return PandaEnvConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
