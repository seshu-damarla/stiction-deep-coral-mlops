"""
Pytest configuration.

This file makes the project root available to Python during testing.

Project structure:

project_root/
    src/
    api/
    artifacts/
    tests/

Without the project root in sys.path, imports such as:

    from src.inference import ...
    from api.main import app

may fail when pytest collects the tests.
"""

import sys
from pathlib import Path


# ------------------------------------------------------------
# Find project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# ------------------------------------------------------------
# Add project root to Python import path
# ------------------------------------------------------------

project_root_string = str(
    PROJECT_ROOT
)


if project_root_string not in sys.path:

    sys.path.insert(
        0,
        project_root_string
    )


print(
    "\nProject root added to Python path:"
)

print(
    PROJECT_ROOT
)