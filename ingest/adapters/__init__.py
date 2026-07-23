"""
Adapter package. Public API + registration point.

Importing this package imports each adapter module, whose @register("<type>")
call wires it into the registry. In M1 the adapter imports below get
uncommented one at a time (git -> docs -> vault -> files), in risk order.
"""
from ingest.adapters.base import (
    make_chunk,
    validate_chunk,
    load_source,
    get_adapter,
    register,
    REQUIRED_KEYS,
)

# --- adapters register on import (added in M1) -------------------------------
from ingest.adapters import git_adapter    # noqa: F401,E402  ("git")
# from ingest.adapters import docs_adapter   # noqa: F401  ("docs")
# from ingest.adapters import vault_adapter  # noqa: F401  ("vault")
# from ingest.adapters import files_adapter  # noqa: F401  ("files")

__all__ = [
    "make_chunk",
    "validate_chunk",
    "load_source",
    "get_adapter",
    "register",
    "REQUIRED_KEYS",
]
