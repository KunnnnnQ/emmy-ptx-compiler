"""Read-only helpers over Kernel IR statement trees."""

from __future__ import annotations

_NESTED = ("body", "else_body")
BARRIERS = ("Sync", "WarpShuffle", "TreeHalve", "CpAsyncWait")


def iter_stmts(body, kind: str):
    """Every statement of type `kind` in `body`, nested ones included."""
    for s in body:
        if type(s).__name__ == kind:
            yield s
        for field in _NESTED:
            inner = getattr(s, field, None)
            if inner:
                yield from iter_stmts(inner, kind)


def has_barrier(body) -> bool:
    """Whether a body synchronizes threads (so every thread must reach it alike)."""
    return any(next(iter_stmts(body, kind), None) is not None for kind in BARRIERS)


def size(body) -> int:
    """Statement count, nested ones included: the cost measure for unrolling."""
    n = 0
    for s in body:
        n += 1
        for field in _NESTED:
            inner = getattr(s, field, None)
            if inner:
                n += size(inner)
    return n


def pretty(s) -> str:
    try:
        return "\n".join(s.pretty())
    except Exception:
        return type(s).__name__


def head(s) -> str:
    """First line of a statement's pretty form: a block's header without its body."""
    lines = pretty(s).splitlines()
    return lines[0] if lines else type(s).__name__


def identity(op) -> float:
    """The neutral element an accumulator of `op` starts from."""
    ident = op.identity
    if ident is None:
        raise ValueError(f"reduction op {op.name!r} has no identity")
    return float(ident)
