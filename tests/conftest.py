"""Session-wide test configuration.

Hypothesis state (example database, unicode tables) is routed into ./.cache/ so
the repository root never accumulates a stray ``.hypothesis/`` directory.
"""

from pathlib import Path

from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

_HYPOTHESIS_HOME = Path(".cache") / "hypothesis"

set_hypothesis_home_dir(_HYPOTHESIS_HOME)

settings.register_profile(
    "default",
    database=DirectoryBasedExampleDatabase(_HYPOTHESIS_HOME / "examples"),
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("default"),
    derandomize=True,
    database=None,
)
settings.load_profile("default")
