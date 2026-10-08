import ast
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_taiji_reliable_mechet_k_eval_8gpu.sh"


def test_formal_k_launcher_has_valid_shell_and_embedded_python():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    script = SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.DOTALL)
    assert len(blocks) == 2
    for block in blocks:
        ast.parse(block)
    assert 'validate_benchmark_source(' in script
    assert 'validate_v2_adapter_manifest(' in script
    assert '--expected-adapter-sha256 "$adapter_sha"' in script
    assert 'torchrun --standalone --nproc_per_node=8' in script
    assert "report['observed_reactions'] != denominator" in script
    assert "report['missing_reactions'] != 0" in script
