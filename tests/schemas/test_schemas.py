"""Property and unit tests for the curation data contracts."""

import json
import math
import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from datasketch import MinHashLSH
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from synthetic_data_factory.schemas import (
    ContractModel,
    CurationMetrics,
    ExportSpecification,
    InstructionSample,
    MinHashConfig,
)
from synthetic_data_factory.schemas._common import (
    INT64_MAX,
    MAX_LONG_TEXT_CHARS,
    MAX_SHORT_TEXT_CHARS,
)
from synthetic_data_factory.schemas.export import (
    MAX_PATH_CHARS,
    MAX_ROW_GROUP_SIZE,
    MIN_ROW_GROUP_SIZE,
)
from synthetic_data_factory.schemas.minhash import (
    MAX_NUM_PERM,
    MAX_SHINGLE_K,
    MIN_NUM_PERM,
    MIN_SHINGLE_K,
)
from synthetic_data_factory.schemas.sample import (
    MAX_DOMAIN_CHARS,
    MAX_METADATA_ENTRIES,
    MAX_METADATA_KEY_CHARS,
)

Payload = dict[str, Any]

# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

texts = st.text(min_size=1, max_size=64).filter(lambda s: bool(s.strip()) and "\x00" not in s)
uuid4s = st.uuids(version=4).map(str)
utc_datetimes = st.datetimes(timezones=st.just(UTC))
domains = st.from_regex(r"[a-z0-9]{1,8}(?:[._-][a-z0-9]{1,8}){0,3}", fullmatch=True)
metadata_keys = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,15}", fullmatch=True)
metadata = st.dictionaries(metadata_keys, texts, max_size=8)

powers_of_two = st.sampled_from(
    [2**n for n in range(MAX_NUM_PERM.bit_length()) if MIN_NUM_PERM <= 2**n <= MAX_NUM_PERM]
)
thresholds = st.floats(min_value=0.0, max_value=1.0, exclude_min=True, allow_nan=False)
# First character excludes "." so "." / ".." segments can never be generated.
path_segments = st.from_regex(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,11}", fullmatch=True)
compressions = st.sampled_from(["zstd", "snappy", "gzip"])


@st.composite
def sample_payloads(draw: st.DrawFn) -> Payload:
    return {
        "sample_id": draw(uuid4s),
        "system_prompt": draw(texts),
        "instruction": draw(texts),
        "response": draw(texts),
        "domain": draw(domains),
        "metadata": draw(metadata),
        "created_at_utc": draw(utc_datetimes),
    }


@st.composite
def funnel_counts(draw: st.DrawFn) -> tuple[int, int, int]:
    raw = draw(st.integers(min_value=0, max_value=INT64_MAX))
    valid = draw(st.integers(min_value=0, max_value=raw))
    deduped = draw(st.integers(min_value=0, max_value=valid))
    return raw, valid, deduped


@st.composite
def parquet_paths(draw: st.DrawFn) -> str:
    dirs = draw(st.lists(path_segments, max_size=4))
    stem = draw(path_segments)
    return "/".join([*dirs, f"{stem}.parquet"])


def _sample(**overrides: Any) -> Payload:
    payload: Payload = {
        "sample_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        "system_prompt": "You are a careful assistant.",
        "instruction": "Add 2 and 2.",
        "response": "4",
        "domain": "math",
        "metadata": {"generator": "seed-v1"},
        "created_at_utc": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    }
    payload.update(overrides)
    return payload


def _metrics(**overrides: Any) -> Payload:
    payload: Payload = {
        "raw_count": 100,
        "schema_valid_count": 90,
        "deduped_count": 80,
        "retention_rate": 0.8,
    }
    payload.update(overrides)
    return payload


def _minhash(**overrides: Any) -> Payload:
    payload: Payload = {"threshold": 0.85, "num_perm": 128, "shingle_k": 5}
    payload.update(overrides)
    return payload


def _export(**overrides: Any) -> Payload:
    payload: Payload = {
        "output_path": "out/curated.parquet",
        "compression": "zstd",
        "row_group_size": 100_000,
    }
    payload.update(overrides)
    return payload


def _assert_rejected(model: type[ContractModel], payload: Payload, field: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        model.model_validate(payload)
    locs = {err["loc"][0] if err["loc"] else "__root__" for err in excinfo.value.errors()}
    assert field in locs, excinfo.value.errors()


# --------------------------------------------------------------------------- #
# Shared contract behaviour                                                   #
# --------------------------------------------------------------------------- #

_VALID_PAYLOADS: list[tuple[type[ContractModel], Payload]] = [
    (InstructionSample, _sample()),
    (CurationMetrics, _metrics()),
    (MinHashConfig, _minhash()),
    (ExportSpecification, _export()),
]
_MODEL_IDS = [model.__name__ for model, _ in _VALID_PAYLOADS]


@pytest.mark.parametrize(("model", "payload"), _VALID_PAYLOADS, ids=_MODEL_IDS)
def test_unknown_fields_are_forbidden(model: type[ContractModel], payload: Payload) -> None:
    _assert_rejected(model, {**payload, "unexpected": 1}, "unexpected")


@pytest.mark.parametrize(("model", "payload"), _VALID_PAYLOADS, ids=_MODEL_IDS)
def test_instances_are_immutable(model: type[ContractModel], payload: Payload) -> None:
    instance = model.model_validate(payload)
    field = next(iter(payload))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(instance, field, payload[field])


@pytest.mark.parametrize(("model", "payload"), _VALID_PAYLOADS, ids=_MODEL_IDS)
def test_json_round_trip_is_lossless(model: type[ContractModel], payload: Payload) -> None:
    instance = model.model_validate(payload)
    assert model.model_validate_json(instance.model_dump_json()) == instance


def test_validation_errors_do_not_echo_input() -> None:
    marker = "sensitive-dataset-text-must-not-reach-logs"
    with pytest.raises(ValidationError) as excinfo:
        InstructionSample.model_validate(_sample(response=f"{marker}\x00", sample_id=marker))
    errors = excinfo.value.errors()
    assert {err["loc"][0] for err in errors} == {"response", "sample_id"}
    assert marker not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# InstructionSample                                                           #
# --------------------------------------------------------------------------- #


@given(sample_payloads())
def test_sample_accepts_every_valid_payload_and_round_trips(payload: Payload) -> None:
    sample = InstructionSample.model_validate(payload)
    assert sample.created_at_utc.tzinfo is UTC
    assert InstructionSample.model_validate_json(sample.model_dump_json()) == sample


def test_sample_metadata_defaults_to_empty() -> None:
    payload = _sample()
    del payload["metadata"]
    assert InstructionSample.model_validate(payload).metadata == {}


@given(st.uuids(version=4))
def test_sample_id_rejects_non_canonical_uuid4_spellings(value: uuid.UUID) -> None:
    for spelling in (str(value).upper(), value.hex, f"{{{value}}}", f"urn:uuid:{value}"):
        _assert_rejected(InstructionSample, _sample(sample_id=spelling), "sample_id")


@given(st.one_of(st.uuids(version=1), st.uuids(version=3), st.uuids(version=5)))
def test_sample_id_rejects_other_uuid_versions(value: uuid.UUID) -> None:
    _assert_rejected(InstructionSample, _sample(sample_id=str(value)), "sample_id")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-uuid",
        "7c9e6679-7425-40de-c44b-e07fc1f90ae7",  # wrong variant nibble
        "7c9e6679-7425-40de-944b-e07fc1f90ae7\n",
        uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7"),
    ],
)
def test_sample_id_rejects_malformed_values(value: object) -> None:
    _assert_rejected(InstructionSample, _sample(sample_id=value), "sample_id")


_TEXT_FIELDS = ["system_prompt", "instruction", "response"]


@pytest.mark.parametrize("field", _TEXT_FIELDS)
@given(blank=st.text(alphabet=" \t\n\r\x0b\x0c\u00a0\u2003\u3000", max_size=16))
def test_text_fields_reject_empty_and_blank(field: str, blank: str) -> None:
    _assert_rejected(InstructionSample, _sample(**{field: blank}), field)


@pytest.mark.parametrize("field", _TEXT_FIELDS)
@pytest.mark.parametrize("bad", ["hello\x00world", "lone \ud800 surrogate", b"bytes", 42, None])
def test_text_fields_reject_unsafe_or_mistyped_values(field: str, bad: object) -> None:
    _assert_rejected(InstructionSample, _sample(**{field: bad}), field)


@pytest.mark.parametrize("field", [*_TEXT_FIELDS, "metadata"])
def test_lone_surrogates_are_rejected_by_pydantic_core(field: str) -> None:
    """Pins the UTF-8 guarantee that _common.py relies on pydantic-core for."""
    text = "lone \ud800 surrogate"
    value: object = {"key": text} if field == "metadata" else text
    with pytest.raises(ValidationError) as excinfo:
        InstructionSample.model_validate(_sample(**{field: value}))
    assert [err["type"] for err in excinfo.value.errors()] == ["string_unicode"]


@pytest.mark.parametrize("field", _TEXT_FIELDS)
def test_text_fields_enforce_length_boundary(field: str) -> None:
    at_limit = "x" * MAX_LONG_TEXT_CHARS
    assert (
        getattr(InstructionSample.model_validate(_sample(**{field: at_limit})), field) == at_limit
    )
    _assert_rejected(InstructionSample, _sample(**{field: at_limit + "x"}), field)


def test_text_fields_preserve_content_verbatim() -> None:
    text = "  leading and trailing whitespace is data  \n"
    assert InstructionSample.model_validate(_sample(response=text)).response == text


@pytest.mark.parametrize(
    "domain",
    ["", "Math", "math ", "-math", "math-", "a..b", "a b", "naïve", "x" * (MAX_DOMAIN_CHARS + 1)],
)
def test_domain_rejects_non_slug_values(domain: str) -> None:
    _assert_rejected(InstructionSample, _sample(domain=domain), "domain")


def test_domain_accepts_slug_at_length_limit() -> None:
    domain = "x" * MAX_DOMAIN_CHARS
    assert InstructionSample.model_validate(_sample(domain=domain)).domain == domain


@pytest.mark.parametrize(
    "bad",
    [
        {"": "v"},
        {" key": "v"},
        {"_key": "v"},
        {"k" * (MAX_METADATA_KEY_CHARS + 1): "v"},
        {"key": ""},
        {"key": "   "},
        {"key": "v" * (MAX_SHORT_TEXT_CHARS + 1)},
        {"key": 1},
        {"key": None},
        {f"k{i}": "v" for i in range(MAX_METADATA_ENTRIES + 1)},
        [("key", "v")],
    ],
)
def test_metadata_rejects_invalid_entries(bad: object) -> None:
    _assert_rejected(InstructionSample, _sample(metadata=bad), "metadata")


def test_metadata_accepts_boundary_sizes() -> None:
    full = {f"k{i}": "v" * MAX_SHORT_TEXT_CHARS for i in range(MAX_METADATA_ENTRIES)}
    full["k" * MAX_METADATA_KEY_CHARS] = full.pop("k0")
    assert InstructionSample.model_validate(_sample(metadata=full)).metadata == full


@given(st.datetimes(timezones=st.none()))
def test_created_at_rejects_naive_datetimes(value: datetime) -> None:
    _assert_rejected(InstructionSample, _sample(created_at_utc=value), "created_at_utc")


@given(
    st.datetimes(),
    st.timedeltas(min_value=timedelta(hours=-23), max_value=timedelta(hours=23)),
)
def test_created_at_rejects_non_utc_offsets(naive: datetime, offset: timedelta) -> None:
    assume(offset != timedelta(0))
    value = naive.replace(tzinfo=timezone(offset))
    _assert_rejected(InstructionSample, _sample(created_at_utc=value), "created_at_utc")


@pytest.mark.parametrize("zone", ["Europe/London", "GMT", "Africa/Abidjan"])
def test_created_at_rejects_zero_offset_zones_that_are_not_utc(zone: str) -> None:
    value = datetime(2026, 1, 15, tzinfo=ZoneInfo(zone))
    assert value.utcoffset() == timedelta(0)
    _assert_rejected(InstructionSample, _sample(created_at_utc=value), "created_at_utc")


@pytest.mark.parametrize(
    "tz", [UTC, timezone(timedelta(0)), ZoneInfo("UTC"), ZoneInfo("Etc/UTC")], ids=repr
)
def test_created_at_normalizes_explicit_utc_spellings(tz: Any) -> None:
    value = datetime(2026, 3, 1, 8, 30, 15, 123456, tzinfo=tz)
    sample = InstructionSample.model_validate(_sample(created_at_utc=value))
    assert sample.created_at_utc.tzinfo is UTC
    assert sample.created_at_utc == value


@pytest.mark.parametrize("value", ["2026-01-01T00:00:00Z", 1_767_225_600, None])
def test_created_at_is_strict_in_python_mode(value: object) -> None:
    _assert_rejected(InstructionSample, _sample(created_at_utc=value), "created_at_utc")


@pytest.mark.parametrize(
    ("stamp", "accepted"),
    [
        ("2026-01-01T00:00:00Z", True),
        ("2026-01-01T00:00:00+00:00", True),
        ("2026-01-01T00:00:00", False),
        ("2026-01-01T01:00:00+01:00", False),
    ],
)
def test_created_at_json_mode_requires_utc(stamp: str, accepted: bool) -> None:
    record = json.loads(InstructionSample.model_validate(_sample()).model_dump_json())
    record["created_at_utc"] = stamp
    document = json.dumps(record)
    if accepted:
        parsed = InstructionSample.model_validate_json(document)
        assert parsed.created_at_utc == datetime(2026, 1, 1, tzinfo=UTC)
        assert parsed.created_at_utc.tzinfo is UTC
    else:
        with pytest.raises(ValidationError):
            InstructionSample.model_validate_json(document)


# --------------------------------------------------------------------------- #
# CurationMetrics                                                             #
# --------------------------------------------------------------------------- #


@given(funnel_counts())
def test_metrics_from_counts_is_always_valid(counts: tuple[int, int, int]) -> None:
    raw, valid, deduped = counts
    metrics = CurationMetrics.from_counts(
        raw_count=raw, schema_valid_count=valid, deduped_count=deduped
    )
    assert 0.0 <= metrics.retention_rate <= 1.0
    assert math.isfinite(metrics.retention_rate)
    assert CurationMetrics.model_validate_json(metrics.model_dump_json()) == metrics


def test_metrics_empty_run_has_zero_retention() -> None:
    metrics = CurationMetrics.from_counts(raw_count=0, schema_valid_count=0, deduped_count=0)
    assert metrics.retention_rate == 0.0
    _assert_rejected(
        CurationMetrics,
        _metrics(raw_count=0, schema_valid_count=0, deduped_count=0, retention_rate=1.0),
        "__root__",
    )


@pytest.mark.parametrize(("deduped", "rate"), [(90, 1), (0, 0)])
def test_metrics_accepts_int_retention_at_bounds(deduped: int, rate: int) -> None:
    payload = _metrics(raw_count=90, deduped_count=deduped, retention_rate=rate)
    assert CurationMetrics.model_validate(payload).retention_rate == float(rate)


@given(funnel_counts(), st.floats(min_value=0.0, max_value=1.0))
def test_metrics_rejects_inconsistent_retention(counts: tuple[int, int, int], rate: float) -> None:
    raw, valid, deduped = counts
    expected = deduped / raw if raw else 0.0
    assume(not math.isclose(rate, expected, rel_tol=1e-9))
    payload = _metrics(
        raw_count=raw, schema_valid_count=valid, deduped_count=deduped, retention_rate=rate
    )
    _assert_rejected(CurationMetrics, payload, "__root__")


@pytest.mark.parametrize(
    "rate", [math.nan, math.inf, -math.inf, -0.01, 1.01, -1e-300, 1.0 + 1e-15, True, "0.8", None]
)
def test_metrics_retention_rejects_non_finite_out_of_range_or_mistyped(rate: object) -> None:
    _assert_rejected(CurationMetrics, _metrics(retention_rate=rate), "retention_rate")


def test_metrics_json_rejects_nan_and_infinity() -> None:
    for literal in ("NaN", "Infinity", "-Infinity"):
        document = (
            f'{{"raw_count":1,"schema_valid_count":1,"deduped_count":1,"retention_rate":{literal}}}'
        )
        with pytest.raises(ValidationError):
            CurationMetrics.model_validate_json(document)


@pytest.mark.parametrize("field", ["raw_count", "schema_valid_count", "deduped_count"])
@pytest.mark.parametrize("bad", [-1, INT64_MAX + 1, 1.0, True, "1", None])
def test_metrics_counts_are_strict_non_negative_int64(field: str, bad: object) -> None:
    _assert_rejected(CurationMetrics, _metrics(**{field: bad}), field)


def test_metrics_counts_accept_int64_max() -> None:
    metrics = CurationMetrics.from_counts(
        raw_count=INT64_MAX, schema_valid_count=INT64_MAX, deduped_count=INT64_MAX
    )
    assert metrics.retention_rate == 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_valid_count": 101, "deduped_count": 80}, "schema_valid_count"),
        ({"schema_valid_count": 70, "deduped_count": 80}, "deduped_count"),
    ],
)
def test_metrics_enforce_funnel_ordering(overrides: Payload, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CurationMetrics.model_validate(_metrics(**overrides))


@given(st.integers(min_value=0, max_value=INT64_MAX - 1), st.data())
def test_metrics_rejects_valid_exceeding_raw(raw: int, data: st.DataObject) -> None:
    valid = data.draw(st.integers(min_value=raw + 1, max_value=INT64_MAX))
    payload = _metrics(raw_count=raw, schema_valid_count=valid, deduped_count=0, retention_rate=0)
    _assert_rejected(CurationMetrics, payload, "__root__")


@given(st.integers(min_value=0, max_value=INT64_MAX - 1), st.data())
def test_metrics_rejects_deduped_exceeding_valid(valid: int, data: st.DataObject) -> None:
    deduped = data.draw(st.integers(min_value=valid + 1, max_value=INT64_MAX))
    payload = _metrics(
        raw_count=INT64_MAX, schema_valid_count=valid, deduped_count=deduped, retention_rate=0.5
    )
    _assert_rejected(CurationMetrics, payload, "__root__")


# --------------------------------------------------------------------------- #
# MinHashConfig                                                               #
# --------------------------------------------------------------------------- #


@given(
    thresholds,
    powers_of_two,
    st.integers(min_value=MIN_SHINGLE_K, max_value=MAX_SHINGLE_K),
)
def test_minhash_accepts_full_valid_domain(threshold: float, num_perm: int, k: int) -> None:
    config = MinHashConfig.model_validate(
        {"threshold": threshold, "num_perm": num_perm, "shingle_k": k}
    )
    assert (config.threshold, config.num_perm, config.shingle_k) == (threshold, num_perm, k)


# datasketch's (bands, rows) optimizer needs >= 2 bands. The highest tunable
# threshold grows with num_perm (~0.863 at 16 perms, ~0.980 at 128, ~0.997 at
# 1024); 0.85 is tunable for every num_perm this contract admits.
UNIVERSALLY_LSH_TUNABLE_THRESHOLD = 0.85


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.floats(min_value=0.0, max_value=UNIVERSALLY_LSH_TUNABLE_THRESHOLD, exclude_min=True),
    st.sampled_from([MIN_NUM_PERM, 32, 64, 128]),
)
def test_minhash_config_is_accepted_by_datasketch_lsh(threshold: float, num_perm: int) -> None:
    config = MinHashConfig(threshold=threshold, num_perm=num_perm, shingle_k=5)
    lsh = MinHashLSH(threshold=config.threshold, num_perm=config.num_perm)
    assert lsh.b >= 2
    assert lsh.b * lsh.r <= config.num_perm


def test_minhash_threshold_one_is_valid_but_not_lsh_tunable() -> None:
    """Known gap, pinned on purpose: the contract admits threshold=1.0 (per spec),
    but datasketch cannot auto-tune LSH bands for it. The dedup stage must route
    such configs to exact-hash deduplication or explicit (b, r) params."""
    config = MinHashConfig(threshold=1.0, num_perm=128, shingle_k=5)
    with pytest.raises(ValueError, match="bands are too small"):
        MinHashLSH(threshold=config.threshold, num_perm=config.num_perm)


@given(st.integers(min_value=MIN_NUM_PERM, max_value=MAX_NUM_PERM))
def test_minhash_num_perm_rejects_non_powers_of_two(num_perm: int) -> None:
    assume(num_perm & (num_perm - 1))
    _assert_rejected(MinHashConfig, _minhash(num_perm=num_perm), "num_perm")


@pytest.mark.parametrize(
    "num_perm", [0, -128, 1, 2, 8, MIN_NUM_PERM - 1, MAX_NUM_PERM * 2, 128.0, True, "128"]
)
def test_minhash_num_perm_rejects_out_of_range_or_mistyped(num_perm: object) -> None:
    _assert_rejected(MinHashConfig, _minhash(num_perm=num_perm), "num_perm")


@pytest.mark.parametrize("num_perm", [MIN_NUM_PERM, MAX_NUM_PERM])
def test_minhash_num_perm_accepts_bounds(num_perm: int) -> None:
    assert MinHashConfig.model_validate(_minhash(num_perm=num_perm)).num_perm == num_perm


@pytest.mark.parametrize(
    "threshold", [0.0, -0.0, -0.1, 1.0 + 1e-12, math.nan, math.inf, -math.inf, True, "0.9", None]
)
def test_minhash_threshold_rejects_values_outside_half_open_unit(threshold: object) -> None:
    _assert_rejected(MinHashConfig, _minhash(threshold=threshold), "threshold")


@pytest.mark.parametrize("threshold", [5e-324, 1.0, 1])
def test_minhash_threshold_accepts_extremes(threshold: float) -> None:
    assert MinHashConfig.model_validate(_minhash(threshold=threshold)).threshold == threshold


@pytest.mark.parametrize("k", [MIN_SHINGLE_K - 1, 0, -2, MAX_SHINGLE_K + 1, 2.0, True, "5", None])
def test_minhash_shingle_k_rejects_invalid(k: object) -> None:
    _assert_rejected(MinHashConfig, _minhash(shingle_k=k), "shingle_k")


# --------------------------------------------------------------------------- #
# ExportSpecification                                                         #
# --------------------------------------------------------------------------- #


@given(
    parquet_paths(),
    compressions,
    st.integers(min_value=MIN_ROW_GROUP_SIZE, max_value=MAX_ROW_GROUP_SIZE),
)
def test_export_accepts_relative_parquet_paths(path: str, compression: str, rows: int) -> None:
    spec = ExportSpecification.model_validate(
        {"output_path": path, "compression": compression, "row_group_size": rows}
    )
    assert spec.output_path == path
    assert (spec.compression, spec.row_group_size) == (compression, rows)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("./out.parquet", "out.parquet"),
        ("a//b/./c.parquet", "a/b/c.parquet"),
        ("data.v2.parquet", "data.v2.parquet"),
    ],
)
def test_export_normalizes_path(raw: str, normalized: str) -> None:
    assert ExportSpecification.model_validate(_export(output_path=raw)).output_path == normalized


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/abs/out.parquet",
        "//server/share/out.parquet",
        "~/out.parquet",
        "~user/out.parquet",
        "C:/out.parquet",
        "c:out.parquet",
        "out\\data.parquet",
        "../out.parquet",
        "a/../../out.parquet",
        "out/..",
        "out.csv",
        "out.PARQUET",
        "out.parquet.bak",
        ".parquet",
        "dir/.parquet",
        "out.parquet/",
        "out",
        " out.parquet",
        "out.parquet ",
        "out\x00.parquet",
        "out\n.parquet",
        "out\x7f.parquet",
        "x" * (MAX_PATH_CHARS - len(".parquet") + 1) + ".parquet",
    ],
)
def test_export_rejects_unsafe_or_non_parquet_paths(path: str) -> None:
    _assert_rejected(ExportSpecification, _export(output_path=path), "output_path")


def test_export_path_accepts_length_limit() -> None:
    path = "x" * (MAX_PATH_CHARS - len(".parquet")) + ".parquet"
    assert ExportSpecification.model_validate(_export(output_path=path)).output_path == path


@given(st.text(min_size=1, max_size=12).filter(lambda s: s not in {"zstd", "snappy", "gzip"}))
def test_export_compression_rejects_unknown_codecs(codec: str) -> None:
    _assert_rejected(ExportSpecification, _export(compression=codec), "compression")


@pytest.mark.parametrize("codec", ["ZSTD", "lz4", "brotli", "none", None, 1])
def test_export_compression_rejects_unsupported_values(codec: object) -> None:
    _assert_rejected(ExportSpecification, _export(compression=codec), "compression")


@pytest.mark.parametrize(
    "rows", [MIN_ROW_GROUP_SIZE - 1, 0, -1, MAX_ROW_GROUP_SIZE + 1, 10_000.0, True, "10000"]
)
def test_export_row_group_size_rejects_invalid(rows: object) -> None:
    _assert_rejected(ExportSpecification, _export(row_group_size=rows), "row_group_size")


@pytest.mark.parametrize("rows", [MIN_ROW_GROUP_SIZE, MAX_ROW_GROUP_SIZE])
def test_export_row_group_size_accepts_bounds(rows: int) -> None:
    assert ExportSpecification.model_validate(_export(row_group_size=rows)).row_group_size == rows
