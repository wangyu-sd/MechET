import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_documentation_integrity.py"
spec = importlib.util.spec_from_file_location("mechet_documentation_integrity", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_documentation_checker_accepts_valid_internal_links_and_commands(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts" / "demo.py").write_text(
        "import argparse\n"
        "def main():\n"
        "    argparse.ArgumentParser().parse_args()\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "demo.yaml").write_text("value: 1\n", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text(
        "# Guide\n\n## Run it\n\n```bash\npython scripts/demo.py --config configs/demo.yaml\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Project\n\n[Guide](docs/guide.md#run-it)\n",
        encoding="utf-8",
    )
    result = module.check(tmp_path)
    assert result["passed"]
    assert result["n_referenced_scripts"] == 1
    assert result["n_referenced_configs"] == 1


def test_documentation_checker_reports_missing_link_anchor_and_script(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Project\n\n"
        "[Missing](docs/missing.md)\n"
        "[Bad anchor](docs/guide.md#not-there)\n\n"
        "```bash\npython scripts/missing.py\n```\n",
        encoding="utf-8",
    )
    result = module.check(tmp_path)
    kinds = {item["type"] for item in result["errors"]}
    assert "missing_internal_link" in kinds
    assert "missing_markdown_anchor" in kinds
    assert "missing_documented_script" in kinds
    assert not result["passed"]
