import subprocess
import json
import pytest


def test_runtime_package_is_immutable_after_source_branch_advances(tmp_path):
    from scripts.prepare_taiji_joint_training import freeze_runtime
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Unit Test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'unit@example.invalid'], check=True)
    (repo/'scripts').mkdir()
    (repo/'scripts'/'example.py').write_text('answer = 1\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-m', 'fixture'], check=True, capture_output=True)
    root = tmp_path/'runtime'
    metadata = freeze_runtime(repo, root)
    (repo/'scripts'/'example.py').write_text('answer = 2\n')
    assert (root/'scripts'/'example.py').read_text() == 'answer = 1\n'
    assert json.loads((root/'runtime_provenance.json').read_text()) == metadata
    with pytest.raises(FileExistsError):
        freeze_runtime(repo, root)
