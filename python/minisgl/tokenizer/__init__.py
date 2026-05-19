from .aliasing import AliasEntry, AliasingGuideTable, build_aliasing_guide

__all__ = [
    "AliasEntry",
    "AliasingGuideTable",
    "build_aliasing_guide",
    "tokenize_worker",
]


def __getattr__(name: str):
    if name == "tokenize_worker":
        from .server import tokenize_worker
        return tokenize_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

