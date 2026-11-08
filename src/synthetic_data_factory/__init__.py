"""Synthetic instruction-dataset curation: contracts, deduplication, and export."""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("synthetic-data-factory")
