"""Pin direct multistep training to the completed, corrected FlowER Stage-II actor."""
from dataclasses import dataclass
import json
from pathlib import Path

from .external_route_data import file_sha

ROOT=Path('/aaa/fionafyang/buddy1/whaleywang/MechET')


@dataclass(frozen=True)
class Stage2ActorPin:
    checkpoint: Path
    adapter_sha256: str
    reference_dir: Path
    reference_manifest_sha256: str
    history_manifest: Path


FLOWER_STAGE2_PIN=Stage2ActorPin(
    ROOT/'outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/models/history/checkpoint-89442',
    '2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453',
    ROOT/'data/pr96_flower_mainonly_source_arrows_v2_20261023_06',
    '2a203ee08df9a6099f66d554797a72b93750b88cdfe760669431f2b665ff083c',
    ROOT/'data/pr96_flower_mainonly_history_v2_20261023_06/manifest.json')


def verify_stage2_actor(parent,*,pin=FLOWER_STAGE2_PIN):
    def require(value,reason):
        if not value:raise ValueError('STAGE2_'+reason)
    def read(path):return json.loads(path.read_text())
    parent=Path(parent).resolve()
    require(parent==pin.checkpoint.resolve(),'CHECKPOINT_PATH_MISMATCH')
    receipt=parent/'pr96_checkpoint_manifest.json'
    require(receipt.is_file(),'HISTORY_CHECKPOINT_MANIFEST_REQUIRED')
    try:
        checkpoint=read(receipt);lineage=checkpoint['lineage']
        require(checkpoint['track']=='flower' and checkpoint['stage']=='history','HISTORY_STAGE_REQUIRED')
        require(checkpoint['epoch']==3. and checkpoint['step']==89442,'COMPLETED_HISTORY_REQUIRED')
        require(Path(checkpoint['best_checkpoint']).resolve()==parent,'SELECTED_CHECKPOINT_REQUIRED')
        require(Path(lineage['reference_dir']).resolve()==pin.reference_dir.resolve(),'CORRECTED_SOURCE_REQUIRED')
        files={}
        for name in ('adapter_model.safetensors','adapter_config.json'):
            path=parent/name;actual=file_sha(path)
            require(actual==checkpoint['files'][name],'ADAPTER_COMPONENT_CHANGED:'+name)
            files[str(path)]=actual
        require(files[str(parent/'adapter_model.safetensors')]==pin.adapter_sha256,'ADAPTER_PIN_CHANGED')
        source_path=pin.reference_dir/'manifest.json';source=read(source_path)
        require(file_sha(source_path)==pin.reference_manifest_sha256,'REFERENCE_MANIFEST_CHANGED')
        require(source['original_moves_preserved'] and source['input_condition']=='main_product_only_v1',
                'CORRECTED_MAIN_PRODUCT_PROTOCOL_REQUIRED')
        history=read(pin.history_manifest)
        require(history['source_manifest_sha256']==pin.reference_manifest_sha256,'HISTORY_REFERENCE_CHANGED')
        require(not history['history_contains_future'] and not history['history_contains_failed_actions'],
                'INVALID_HISTORY_CONTENT')
        adapter_path=parent.parent/'adapter_manifest.json';adapter=read(adapter_path)
        require(adapter['pr96']['full_epochs_completed']==3. and
                Path(adapter['pr96']['best_checkpoint']).resolve()==parent,'INCOMPLETE_OR_UNSELECTED_HISTORY')
        require(adapter['adapter_sha256']==pin.adapter_sha256,'SELECTED_ADAPTER_CHANGED')
        data_path=parent.parent/'data_contract.json';data=read(data_path)
        require(data['artifact_status']['training_allowed'] is True,'DEPLOYED_TRAINING_GATES_REQUIRED')
        preflight_path=Path(lineage['preflight_manifest']).resolve();preflight=read(preflight_path)
        require(file_sha(preflight_path)==lineage['preflight_manifest_sha256']==
                data['artifact_status']['deployed_preflight_sha256'],'DEPLOYED_PREFLIGHT_CHANGED')
        ranks=preflight['ranks']
        require(preflight['track']=='flower' and sorted(r['rank'] for r in ranks)==list(range(8)),
                'DEPLOYED_PREFLIGHT_RANKS_REQUIRED')
        gates=('finite_nonzero_gradients','parameters_updated','ddp_gradient_synchronization',
               'assistant_mask_verified','tool_prefix_parity','generation_completed')
        require(all(all(r[k] is True for k in gates) and
                    r['performance_layout_probe']['finite_backward'] is True for r in ranks),
                'DEPLOYED_PREFLIGHT_GATES_FAILED')
        tokens_path=parent.parent.parent.parent/'tokens/history/manifest.json';tokens=read(tokens_path)
        require(file_sha(tokens_path)==lineage['token_cache_manifest_sha256']==
                data['artifact_status']['token_cache_manifest_sha256'],'TOKEN_MANIFEST_CHANGED')
        require(lineage['train_file_sha256']==data['train_file_sha256']==tokens['sources']['train']['sha256']==
                history['reports']['train']['sha256'],'TRAINING_SOURCE_CHANGED')
        for path in (receipt,source_path,pin.history_manifest,adapter_path,data_path,preflight_path,tokens_path):
            files[str(path)]=file_sha(path)
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('STAGE2_INVALID_PROVENANCE:'+str(exc)) from exc
    return dict(kind='stage2_history_sft',checkpoint=str(parent),adapter_sha256=pin.adapter_sha256,
                corrected_reference=str(pin.reference_dir),reference_manifest_sha256=pin.reference_manifest_sha256,
                history_epochs_completed=3.,train_file_sha256=data['train_file_sha256'],files_sha256=files)


def verify_actor_binding(cfg,*,pin=FLOWER_STAGE2_PIN):
    if 'starting_actor_condition' not in cfg and 'actor_lineage' not in cfg:return None
    if cfg.get('starting_actor_condition')!='corrected_flower_stage2_history':
        raise ValueError('STAGE2_STARTING_CONDITION_REQUIRED')
    actual=verify_stage2_actor(cfg['parent'],pin=pin)
    public={k:v for k,v in actual.items() if k!='files_sha256'}
    if cfg.get('parent_sha256')!=actual['adapter_sha256'] or cfg.get('actor_lineage')!=public:
        raise ValueError('STAGE2_CONTRACT_PROVENANCE_CHANGED')
    if any(cfg['files_sha256'].get(p)!=h for p,h in actual['files_sha256'].items()):
        raise ValueError('STAGE2_PROVENANCE_FILE_BINDING_MISSING')
    return actual
