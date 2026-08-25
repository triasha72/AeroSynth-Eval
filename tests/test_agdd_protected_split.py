import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_agdd_protected_split.py"
SPEC = importlib.util.spec_from_file_location("freeze_agdd_protected_split", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
partition_ids = MODULE.partition_ids


def test_partition_is_deterministic_disjoint_and_complete() -> None:
    sample_ids = [f"sample-{index}" for index in range(22)]

    selection, test = partition_ids(sample_ids)

    assert selection == partition_ids(list(reversed(sample_ids)))[0]
    assert len(selection) == 11
    assert len(test) == 11
    assert set(selection).isdisjoint(test)
    assert set(selection) | set(test) == set(sample_ids)
