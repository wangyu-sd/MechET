"""CPU memory over complete execution scopes and hash-bound verifier receipts.

Receipts must be produced by the independent offline verifier. This API checks
provenance and eligibility; it does not independently prove chemical feasibility.
It never replaces the full route event archive or executes recalled atom handles.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

SHARED = Path('/aaa/fionafyang/buddy1')
_SCOPE_FIELDS = {'state','target','route_root','next_map','imported','visited',
    'accepted_action_types','ancestors','stock_version','executor_version','occurrence',
    'event_history_sha256'}


def scope_key(scope):
    if not isinstance(scope, dict) or not _SCOPE_FIELDS.issubset(scope):
        raise ValueError('MEMORY_COMPLETE_EXECUTION_SCOPE_REQUIRED')
    for key in ('state','target','route_root','stock_version','executor_version','occurrence',
                'event_history_sha256'):
        if not isinstance(scope[key], str) or not scope[key]:
            raise ValueError('MEMORY_SCOPE_FIELD:'+key)
    for key in ('visited','ancestors','accepted_action_types'):
        if not isinstance(scope[key], list) or any(not isinstance(v,str) for v in scope[key]):
            raise ValueError('MEMORY_SCOPE_FIELD:'+key)
    if (type(scope['next_map']) is not int or scope['next_map']<1
            or not isinstance(scope['imported'],dict)):
        raise ValueError('MEMORY_SCOPE_ATOM_AND_IMPORT_STATE')
    return hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def _training(split):
    if split != 'training':
        raise ValueError('MEMORY_LEARNING_REQUIRES_TRAIN')


def _evidence(pointer):
    if (not isinstance(pointer,dict) or set(pointer)!={'path','sha256'}
            or not isinstance(pointer['sha256'],str) or len(pointer['sha256'])!=64):
        raise ValueError('MEMORY_HASH_BOUND_EVIDENCE_REQUIRED')
    path=Path(pointer['path']).resolve()
    if not path.is_relative_to(SHARED):
        raise ValueError('MEMORY_SHARED_EVIDENCE_REQUIRED')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pointer['sha256']:
        raise ValueError('MEMORY_EVIDENCE_HASH_MISMATCH')
    data=json.loads(raw)
    _training(data.get('split'))
    if (data.get('episode_complete') is not True or data.get('resource_censored') is not False
            or type(data.get('success')) is not bool
            or not isinstance(data.get('receipt_id'),str) or not data['receipt_id']
            or (data.get('signal'),data.get('verification')) not in {
                ('executor_recovery','executor_replayed'),
                ('reference_route','reference_and_stock_verified')}):
        raise ValueError('MEMORY_INELIGIBLE_VERIFIER_EVIDENCE')
    return data


def _bound_evidence(pointer, uid, key, content):
    proof=_evidence(pointer)
    digest=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()
    if (proof.get('memory_id')!=uid or proof.get('scope_key')!=key
            or proof.get('content_sha256')!=digest):
        raise ValueError('MEMORY_EVIDENCE_BINDING_MISMATCH')
    return proof


class ExperienceMemory:
    def __init__(self, snapshot=None):
        self.entries={}
        if snapshot is not None:
            if snapshot.get('schema')!='route_experience_memory_v2':
                raise ValueError('MEMORY_SCHEMA_MISMATCH')
            for entry in snapshot['entries']:
                item=deepcopy(entry)
                if item['scope_key']!=scope_key(item['scope']) or item['id'] in self.entries:
                    raise ValueError('MEMORY_SNAPSHOT_SCOPE_OR_ID')
                if not item['evidence']:
                    raise ValueError('MEMORY_EVIDENCE_REQUIRED')
                for pointer in item['evidence']:
                    if _bound_evidence(pointer,item['id'],item['scope_key'],item['content'])['signal']!=item['signal']:
                        raise ValueError('MEMORY_EVIDENCE_SIGNAL')
                for uid,record in item['feedback'].items():
                    proof=_bound_evidence(record['evidence'],item['id'],item['scope_key'],item['content'])
                    if (proof['receipt_id']!=uid or proof['success']!=record['success']
                            or proof['signal']!=item['signal']):
                        raise ValueError('MEMORY_SNAPSHOT_FEEDBACK')
                successes=sum(v['success'] for v in item['feedback'].values())
                failures=len(item['feedback'])-successes
                utility=successes/len(item['feedback']) if item['feedback'] else .5
                if (item['successes'],item['failures'],item['utility'])!=(successes,failures,utility):
                    raise ValueError('MEMORY_SNAPSHOT_COUNTS')
                self.entries[item['id']]=item

    def add(self, uid, scope, content, *, evidence=None, split='training', evidence_ids=None):
        _training(split)
        key=scope_key(scope)
        if not isinstance(evidence,list) or not evidence:
            raise ValueError('MEMORY_HASH_BOUND_EVIDENCE_REQUIRED')
        proofs=[_bound_evidence(p,uid,key,content) for p in evidence]
        if len({p['signal'] for p in proofs})!=1:
            raise ValueError('MEMORY_EVIDENCE_SIGNAL')
        if not isinstance(uid,str) or not uid or uid in self.entries:
            raise ValueError('MEMORY_DUPLICATE_OR_INVALID_ID')
        content=json.loads(json.dumps(content,allow_nan=False))
        self.entries[uid]=dict(id=uid,scope=deepcopy(scope),scope_key=key,content=content,
            evidence=deepcopy(evidence),signal=proofs[0]['signal'],
            feedback={},successes=0,failures=0,utility=.5)

    def feedback(self, uid, *, success, evidence=None, split='training',
                 resource_censored=False, evidence_id=None):
        _training(split)
        if type(success) is not bool or type(resource_censored) is not bool:
            raise ValueError('MEMORY_FEEDBACK_BOOL_REQUIRED')
        if resource_censored:
            return False
        item=self.entries[uid]
        proof=_bound_evidence(evidence,uid,item['scope_key'],item['content'])
        if proof['success']!=success or proof['signal']!=item['signal']:
            raise ValueError('MEMORY_CONFLICTING_FEEDBACK_EVIDENCE')
        rid=proof['receipt_id']
        record=dict(success=success,evidence=deepcopy(evidence))
        if rid in item['feedback']:
            if item['feedback'][rid]!=record:
                raise ValueError('MEMORY_CONFLICTING_FEEDBACK_RECEIPT')
            return False
        item['feedback'][rid]=record
        item['successes']+=int(success);item['failures']+=int(not success)
        item['utility']=item['successes']/len(item['feedback'])
        return True

    def retrieve(self, scope, *, limit=5):
        if type(limit) is not int or limit<1:
            raise ValueError('MEMORY_POSITIVE_LIMIT_REQUIRED')
        key=scope_key(scope)
        items=[v for v in self.entries.values() if v['scope_key']==key]
        return deepcopy(sorted(items,key=lambda v:(-v['utility'],v['id']))[:limit])

    def snapshot(self):
        return dict(schema='route_experience_memory_v2',
                    entries=deepcopy([self.entries[k] for k in sorted(self.entries)]))
