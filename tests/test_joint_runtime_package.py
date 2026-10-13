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


def test_runtime_freezes_external_crn_source_and_hashes_it(tmp_path):
    from scripts.prepare_taiji_joint_training import freeze_runtime
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git','init',str(repo)], check=True, capture_output=True)
    subprocess.run(['git','-C',str(repo),'config','user.name','Unit Test'], check=True)
    subprocess.run(['git','-C',str(repo),'config','user.email','unit@example.invalid'], check=True)
    (repo/'scripts').mkdir()
    (repo/'scripts'/'test.py').write_text('pass\n')
    subprocess.run(['git','-C',str(repo),'add','.'], check=True)
    subprocess.run(['git','-C',str(repo),'commit','-m','fixture'], check=True,capture_output=True)
    crn = tmp_path/'crn'
    (crn/'src'/'crn_jepa').mkdir(parents=True)
    (crn/'src'/'crn_jepa'/'test.py').write_text('version = 1\n')
    root = tmp_path/'runtime'
    metadata = freeze_runtime(repo, root, crn_root=crn)
    (crn/'src'/'crn_jepa'/'test.py').write_text('version = 2\n')
    name = 'crn_snapshot/src/crn_jepa/test.py'
    assert (root/name).read_text() == 'version = 1\n'
    assert name in metadata['sha256']
