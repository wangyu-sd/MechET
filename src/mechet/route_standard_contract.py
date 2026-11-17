"""Full source-split contract, separate from historical curated D1 guards."""
import json
from pathlib import Path
import sqlite3

from .external_route_data import file_sha,iter_records

COUNTS={'reaction-holdout-n1-n5':(156723,8250),'route-holdout-n1-n5':(160670,8456)}


def dataset_bindings(dataset,expected_manifest_sha256,*,expected_counts=None):
    dataset=Path(dataset).resolve();path=dataset/'manifest.json'
    if file_sha(path)!=expected_manifest_sha256:raise ValueError('STANDARD_MANIFEST_HASH_CHANGED')
    manifest=json.loads(path.read_text())
    if (manifest.get('schema')!='external_target_only_route_standard_v1' or
            manifest.get('policy_input')!='target_main_product_only_v1' or
            manifest.get('reference_action_labels') is not False):
        raise ValueError('STANDARD_TARGET_ONLY_DATASET_REQUIRED')
    if (manifest.get('source_rows_filtered')!=0 or manifest.get('excluded')!={} or
            manifest.get('source_split_policy')!='preserve_source_training_validation_without_reassignment' or
            manifest['stock_protocol'].get('stock_pruning') is not False or
            manifest['stock_protocol'].get('stock_closure_filter') is not False):
        raise ValueError('STANDARD_SOURCE_FILTER_OR_REASSIGNMENT_FORBIDDEN')
    counts=expected_counts or COUNTS[manifest['artifact']]
    if (manifest['training_targets'],manifest['validation_targets'])!=tuple(counts):
        raise ValueError('STANDARD_FULL_DENOMINATOR_CHANGED')
    files={str(path):expected_manifest_sha256}
    for name,sha in manifest['artifacts'].items():
        if Path(name).name!=name or file_sha(dataset/name)!=sha:
            raise ValueError('STANDARD_ARTIFACT_HASH_CHANGED:'+name)
        files[str(dataset/name)]=sha
    for name in ('protected_identities.json','training_protected_identities.json','validation_protected_identities.json'):
        if json.loads((dataset/name).read_text())!=[]:
            raise ValueError('STANDARD_HIDDEN_CENSORING_FORBIDDEN')
    with sqlite3.connect((dataset/'references.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        for split,expected in zip(('training','validation'),counts):
            seen=set()
            for row in iter_records(dataset/(split+'.jsonl')):
                if set(row)!={'task_id','target'} or row['task_id'] in seen:
                    raise ValueError('STANDARD_PUBLIC_TASK_SCHEMA_OR_DUPLICATE')
                seen.add(row['task_id'])
            if len(seen)!=expected or db.execute('SELECT count(*) FROM tasks WHERE split=?',(split,)).fetchone()[0]!=expected:
                raise ValueError('STANDARD_PUBLIC_DENOMINATOR_CHANGED')
    for record in manifest['source_files'].values():
        if file_sha(record['path'])!=record['sha256']:raise ValueError('STANDARD_SOURCE_HASH_CHANGED')
        files[record['path']]=record['sha256']
    for path_key,sha_key in [('source_manifest','source_manifest_sha256'),('release_checksums','release_checksums_sha256')]:
        if file_sha(manifest[path_key])!=manifest[sha_key]:raise ValueError('STANDARD_RELEASE_HASH_CHANGED')
        files[manifest[path_key]]=manifest[sha_key]
    return dict(dataset=str(dataset),dataset_manifest_sha256=expected_manifest_sha256,
        train_targets=counts[0],validation_targets=counts[1],files_sha256=files,
        standard_data=dict(version='complete_source_split_v1',release=manifest['release'],
            artifact=manifest['artifact'],source_manifest_sha256=manifest['source_manifest_sha256'],
            stock_protocol=manifest['stock_protocol'],task_identity=manifest['task_identity'],
            known_cross_split_audit=manifest['cross_split_audit'],
            benchmark_overlap=manifest['benchmark_overlap'],policy_input=manifest['policy_input'],
            sft_exposure='reaction_holdout_does_not_erase_frozen_FlowER_pretraining'))
