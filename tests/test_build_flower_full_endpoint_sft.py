from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_flower_full_endpoint_sft import iter_trajectory_ids, split_endpoint_roles
from build_flower_retro import build_split as build_flower_retro_split
from build_mechet_sft import _build_row, iter_flower_groups
from mechet.mech_graph import build_mechanism_graph


def test_noncontiguous_trajectory_ids_are_yielded_once(tmp_path: Path):
    source = tmp_path / "test.txt"
    source.write_text(
        "[CH3:1]>>[CH3:1]|1\n"
        "[NH2:2]>>[NH2:2]|2\n"
        "[CH3:1]>>[CH3:1]|1\n",
        encoding="utf-8",
    )
    assert list(iter_trajectory_ids(source)) == ["1", "2"]

    groups = list(iter_flower_groups(source))
    assert [identifier for identifier, _steps in groups] == ["1", "2"]
    assert [len(steps) for _identifier, steps in groups] == [2, 1]


def test_group_limit_still_collects_late_rows(tmp_path: Path):
    source = tmp_path / "test.txt"
    source.write_text(
        "[CH3:1]>>[CH2:1]|1\n"
        "[NH2:2]>>[NH:2]|2\n"
        "[CH2:1]>>[CH2:1]|1\n",
        encoding="utf-8",
    )
    groups = list(iter_flower_groups(source, limit=1))
    assert len(groups) == 1
    assert groups[0][0] == "1"
    assert len(groups[0][1]) == 2


def test_mechet_builder_uses_current_sft_converter_signature():
    graph = build_mechanism_graph(
        "toy",
        [
            ("[CH3:1][Br:2].[OH-:3]", "[CH3:1][OH:3].[Br-:2]"),
            ("[CH3:1][OH:3].[Br-:2]", "[CH3:1][OH:3].[Br-:2]"),
        ],
    )
    assert graph is not None
    row = _build_row(graph, source_split="test")
    assert row is not None
    assert row["task_type"] == "mech_et_cot_retro"


def test_endpoint_roles_use_main_product_atom_maps():
    structural, auxiliary, mapped = split_endpoint_roles(
        "[CH3:1][Br:2].[Na+:3].[OH2:4]", "[CH3:1][OH:4]"
    )
    assert mapped is True
    assert set(structural.split(".")) == {"[CH3:1][Br:2]", "[OH2:4]"}
    assert auxiliary == ["[Na+:3]"]


def test_flower_retro_is_first_lhs_to_last_rhs_in_first_id_order(tmp_path: Path):
    flower_root = tmp_path / "flower_new_dataset"
    flower_root.mkdir()
    (flower_root / "test.txt").write_text(
        "A>>B|10\n"
        "X>>Y|20\n"
        "B>>C|10\n"
        "Y>>Z|20\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "flower_retro"

    report = build_flower_retro_split(
        flower_root,
        output_dir,
        "test",
        require_canonical=False,
    )

    assert report["rows"] == 2
    assert (output_dir / "test.txt").read_text(encoding="utf-8") == (
        "A>>C\nX>>Z\n"
    )
