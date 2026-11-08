"""Columnar transport: canonical Arrow schema, atomic Parquet sink, streaming reader."""

from synthetic_data_factory.transport.parquet_sink import (
    AtomicParquetSink,
    ExportReport,
    ExportValidationError,
    SinkState,
)
from synthetic_data_factory.transport.reader import (
    ReaderStats,
    RejectReason,
    StreamingDatasetReader,
    parse_record,
)
from synthetic_data_factory.transport.schema import (
    INSTRUCTION_SAMPLE_SCHEMA,
    SchemaMismatchError,
    conform,
    iter_samples,
    polars_to_samples,
    polars_to_table,
    samples_to_polars,
    samples_to_record_batch,
    samples_to_table,
    schema_matches,
    table_to_polars,
    table_to_samples,
)

__all__ = [
    "INSTRUCTION_SAMPLE_SCHEMA",
    "AtomicParquetSink",
    "ExportReport",
    "ExportValidationError",
    "ReaderStats",
    "RejectReason",
    "SchemaMismatchError",
    "SinkState",
    "StreamingDatasetReader",
    "conform",
    "iter_samples",
    "parse_record",
    "polars_to_samples",
    "polars_to_table",
    "samples_to_polars",
    "samples_to_record_batch",
    "samples_to_table",
    "schema_matches",
    "table_to_polars",
    "table_to_samples",
]
