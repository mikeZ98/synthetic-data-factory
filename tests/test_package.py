"""Package-level invariants."""

from importlib.metadata import version

import synthetic_data_factory


def test_version_is_sourced_from_distribution_metadata() -> None:
    assert synthetic_data_factory.__version__ == version("synthetic-data-factory")
    assert synthetic_data_factory.__version__ == "0.1.0"
