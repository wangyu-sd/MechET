import pytest

from mechet.chemical_runtime import rdkit_version_tuple


def test_rdkit_version_tuple_handles_calendar_and_suffix_versions():
    assert rdkit_version_tuple("2026.03.4") == (2026, 3, 4)
    assert rdkit_version_tuple("2026.03.4.dev1") == (2026, 3, 4)
    assert rdkit_version_tuple("2024.09.6") < (2026, 3, 4)


def test_rdkit_version_tuple_pads_short_versions():
    assert rdkit_version_tuple("2026.3") == (2026, 3, 0)
    with pytest.raises(ValueError):
        rdkit_version_tuple("not-a-version")
