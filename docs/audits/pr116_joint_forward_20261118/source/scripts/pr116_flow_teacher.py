"""Frozen FlowER fixed-M path sampler. One GPU, read-only public inputs, no route oracle."""
import argparse, hashlib, json, os, sys
from pathlib import Path

def verify(path,expected):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    if h.hexdigest()!=expected:raise ValueError('FlowER source/weight binding changed: '+str(path))

def run(cfg, queries):
    official=Path(cfg['official_code'])
    verify(cfg['checkpoint'],cfg['checkpoint_sha256'])
    for name,digest in cfg['official_files'].items():verify(official/name,digest)
    # Upstream settings.Args resolves these at import time, including when unpickling a checkpoint.
    os.environ.update(DATA_NAME='flower_new_dataset',EXP_NAME='best_large_hyperparam',EMB_DIM='256',
                      RBF_HIGH='12',RBF_GAP='0.1',SIGMA='0.15',SCALE='1',NUM_GPUS_PER_NODE='1')
    sys.path.insert(0,str(official))
    import torch
    from rdkit import Chem
    from train import init_model,init_loader
    from utils.data_utils import ReactionDataset,BEmatrix_to_mol,ps
    from eval_multiGPU import predict_batch,custom_round
    from settings import Args
    if torch.cuda.device_count()!=1:raise ValueError('forward worker requires exactly one visible GPU')
    device=torch.device('cuda:0')
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    args=checkpoint['args']
    args.local_rank=-1;args.device=device;args.load_from=None;args.model_path=cfg['work_dir']
    model,flow,_=init_model(args)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()},strict=True)
    model.eval();flow.eval()
    infer=Args();infer.device=device;infer.local_rank=-1;infer.num_workers=0;infer.sample_size=1
    def matches(mol,q):
        for fragment in Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True):
            heavy=sorted(a.GetAtomMapNum() for a in fragment.GetAtoms() if a.GetAtomicNum()>1)
            if heavy!=q['target_heavy_teacher_maps']:continue
            for a in fragment.GetAtoms():a.SetAtomMapNum(0)
            if Chem.MolToSmiles(Chem.RemoveHs(fragment),isomericSmiles=True)==q['target_product']:return True
        return False
    def score(q):
        # Each physical batch contains independent paths of the same query; a query's
        # seed and draw order are invariant to cache hits, rank order and retries.
        M=cfg['samples']
        seed=int(hashlib.sha256(json.dumps([cfg['seed'],q['teacher_reactants_mapped']],ensure_ascii=True).encode()).hexdigest()[:15],16)
        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        states=[q['teacher_reactants_mapped']]*M
        hit=0
        for depth in range(cfg['max_elementary_steps']):
            if not states:break
            try:
                ds=ReactionDataset(infer,states,parallel=False,reactant_only=True)
            except AssertionError as exc:
                if str(exc)=='Empty Data':return None
                raise
            if len(ds.src_smis)!=len(states):return None
            loader=init_loader(infer,ds,batch_size=cfg['token_batch'],shuffle=False,epoch=None,use_sort=False)
            next_states=[]
            for i,b in enumerate(loader):
                b.to(device)
                matrices=custom_round(predict_batch(infer,i,b,model,flow,1)[-1])
                for j,smiles in enumerate(b.src_smiles_list):
                    n=int(b.src_lens[j].item());mat=matrices[j,:n,:n].cpu().numpy()
                    try:
                        initial=b.src_matrices[j,:n,:n]
                        if not (mat.sum()==initial.sum().item()):continue
                        product=BEmatrix_to_mol(Chem.MolFromSmiles(smiles,ps),mat)
                        Chem.SanitizeMol(product)
                        end=Chem.MolToSmiles(product,isomericSmiles=True,allHsExplicit=True)
                        src=Chem.MolToSmiles(Chem.MolFromSmiles(smiles,ps),isomericSmiles=True,allHsExplicit=True)
                        if end==src:
                            hit+=int(matches(product,q))
                        else:next_states.append(end)
                    except (ValueError,RuntimeError,KeyError,OverflowError):
                        # A completed stochastic draw with a non-chemical output contributes no hit.
                        continue
            states=next_states
        return hit/M
    with torch.inference_mode():return [score(q) for q in queries]

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config');parser.add_argument('queries');parser.add_argument('result')
    a=parser.parse_args()
    scores=run(json.loads(Path(a.config).read_bytes()),json.loads(Path(a.queries).read_bytes()))
    Path(a.result).write_text(json.dumps(scores,allow_nan=False))
