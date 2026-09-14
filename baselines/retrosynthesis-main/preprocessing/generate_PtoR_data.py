import numpy as np
import pandas as pd
import argparse
import hashlib
import json
import os
import re
import random
import textdistance
import multiprocessing

from rdkit import Chem
from tqdm import tqdm


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smi_tokenizer(smi):
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def get_cano_map_number(smi,root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi,root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for i,index in enumerate(cano2atommapIdx)]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def get_root_id(mol,root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root
    # root = -1
    # for i, atom in enumerate(mol.GetAtoms()):
    #     if atom.GetAtomMapNum() == root_map_number:
    #         return i


"""single process version"""
# def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None, reagent=False,root_aligned=True,character=False):
#     """
#     preprocess reaction data to extract graph adjacency matrix and features
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     total = 0
#     src_data = []
#     tgt_data = []
#     differ = 0
#     skip_dict = {
#         'invalid_p':0,
#         'invalid_r':0,
#         'small_p':0,
#         'small_r':0,
#         'error_mapping':0,
#         'error_mapping_p':0,
#         'empty_p':0,
#         'empty_r':0,
#     }
#     pt = re.compile(r':(\d+)]')
#     # cano_dis = []
#     # root_dis = []
#     pbar = tqdm(range(len(reactants)))
#     for index in pbar:
#         pbar.set_description(f"Data Size:{len(src_data)}")
#         product = products[index]
#         reactant = reactants[index]
#
#         if len(src_data) % augmentation != 0:
#             print(index)
#             exit(1)
#
#         # if index < 2281:
#         #     continue
#
#         pro_mol = Chem.MolFromSmiles(product)
#         rea_mol = Chem.MolFromSmiles(reactant)
#
#         """checking data quality"""
#         rids = sorted(re.findall(pt, reactant))
#         pids = sorted(re.findall(pt, product))
#         if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
#             skip_dict['error_mapping'] += 1
#             continue
#         if len(set(rids)) != len(rids) :  # mapping is not 1:1
#             skip_dict['error_mapping'] += 1
#             continue
#         if len(set(pids)) != len(pids) :  # mapping is not 1:1
#             skip_dict['error_mapping'] += 1
#             continue
#         if "" == product:
#             skip_dict['empty_p'] += 1
#             continue
#         if "" == reactant:
#             skip_dict['empty_r'] += 1
#             continue
#         if rea_mol is None:
#             skip_dict['invalid_r'] += 1
#             continue
#         if len(rea_mol.GetAtoms()) < 5:
#             skip_dict['small_r'] += 1
#             continue
#         if pro_mol is None:
#             skip_dict['invalid_p'] += 1
#             continue
#         if len(pro_mol.GetAtoms()) == 1:
#             skip_dict['small_p'] += 1
#             continue
#         if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
#             skip_dict['error_mapping_p'] += 1
#             continue
#         """finishing checking data quality"""
#
#         pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
#         # if re.findall(r"(?<=:)\d+", product) != re.findall(pt,product):
#         #     print(index)
#         #     print(pro_atom_map_numbers)
#         #     print(re.findall(pt,product))
#         #     exit(1)
#         reactant = reactant.split(".")
#         product_roots = [-1]
#         # reversable = len(reactant) > 1
#         reversable = False # no shuffle
#
#         max_times = len(pro_atom_map_numbers)
#         times = min(augmentation, max_times)
#         if times < augmentation: # times = max_times
#             product_roots.extend(pro_atom_map_numbers)
#             product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
#         else: # times = augmentation
#             while len(product_roots) < times:
#                 product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
#                 # pro_atom_map_numbers.remove(product_roots[-1])
#                 if product_roots[-1] in product_roots[:-1]:
#                     product_roots.pop()
#         times = len(product_roots)
#         assert times == augmentation
#         if reversable:
#             times = int(times / 2)
#         for k in range(times):
#             pro_root_atom_map = product_roots[k]
#             pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
#             cano_atom_map = get_cano_map_number(product,root=pro_root)
#             if cano_atom_map is None:
#                 print(f"Product {index} Failed to find Canonical Mol with Atom MapNumber")
#                 continue
#             pro_smi = clear_map_canonical_smiles(product,canonical=True,root=pro_root)
#             # root_map_number = cano_atom_map[0]
#
#             # rea_mols = [Chem.MolFromSmiles(rea) for rea in reactant]
#             # rea_atom_number = [len(mol.GetAtoms()) for mol in rea_mols]
#             # sorted_reactants = sorted(list(zip(reactant, rea_mols,rea_atom_number)),key=lambda x:x[2],reverse=True)
#             # reactant = [i[0] for i in sorted_reactants]
#
#             aligned_reactants = []
#             aligned_reactants_order = []
#             rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
#             used_indices = []
#             for i, rea_map_number in enumerate(rea_atom_map_numbers):
#                 for j, map_number in enumerate(cano_atom_map):
#                     # select mapping reactans
#                     if map_number in rea_map_number:
#                         rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]),root_map_number=map_number)
#                         rea_smi = clear_map_canonical_smiles(reactant[i],canonical=True,root=rea_root)
#                         aligned_reactants.append(rea_smi)
#                         aligned_reactants_order.append(j)
#                         used_indices.append(i)
#                         break
#             sorted_reactants = sorted(list(zip(aligned_reactants,aligned_reactants_order)),key=lambda x:x[1])
#             if not [item[0] for item in sorted_reactants] == aligned_reactants:
#                 print(index)
#             aligned_reactants = [item[0] for item in sorted_reactants]
#             if reagent:
#                 for i in range(len(reactant)):
#                     if i not in used_indices:
#                         aligned_reactants.append(clear_map_canonical_smiles(reactant[i]))
#             # aligned_reactants_order = [item[1] for item in sorted_reactants]
#
#             # product_smi = clear_map_canonical_smiles(pro_smi)
#             reactant_smi = ".".join(aligned_reactants)
#             # if reactant_smi != clear_map_canonical_smiles(reactant):
#             #     differ += 1
#             # product_tokens = smi_tokenizer(product_smi)
#             product_tokens = smi_tokenizer(pro_smi)
#             reactant_tokens = smi_tokenizer(reactant_smi)
#             src_data.append(product_tokens)
#             tgt_data.append(reactant_tokens)
#             # if Chem.MolFromSmiles(pro_smi).GetAtomWithIdx(0).GetSymbol() != Chem.MolFromSmiles(reactant_smi).GetAtomWithIdx(0).GetSymbol():
#             #     print(index)
#             #     print(product_tokens)
#             #     print(reactant_tokens)
#
#             # cano_dis.append(Levenshtein.distance(pro_smi,clear_map_canonical_smiles(reactant_smi)))
#             # root_dis.append(Levenshtein.distance(pro_smi,reactant_smi))
#             if reversable:
#                 aligned_reactants.reverse()
#                 reactant_smi = ".".join(aligned_reactants)
#                 product_tokens = smi_tokenizer(pro_smi)
#                 reactant_tokens = smi_tokenizer(reactant_smi)
#                 src_data.append(product_tokens)
#                 tgt_data.append(reactant_tokens)
#
#     # print(f"Cano Mean {np.mean(cano_dis)} Median {np.median(cano_dis)}")
#     # print(f"Root Mean {np.mean(root_dis)} Median {np.median(root_dis)}")
#     print(differ/len(src_data))
#
#     print('size', len(src_data))
#     for key,value in skip_dict.items():
#         print(f"{key}:{value},{value/len(reactants)}")
#     with open(
#             os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
#         for src in src_data:
#             f.write('{}\n'.format(src))
#
#     with open(
#             os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
#         for tgt in tgt_data:
#             f.write('{}\n'.format(tgt))
#     return src_data,tgt_data

"""multiprocess"""
def _stable_ids_sha256(stable_ids):
    digest = hashlib.sha256()
    for stable_id in stable_ids:
        digest.update(str(stable_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(
    save_dir,
    reactants,
    products,
    set_name,
    augmentation=1,
    reaction_types=None,
    root_aligned=True,
    character=False,
    processes=-1,
    stable_ids=None,
    row_indices=None,
    seed=33,
    strict_coverage=False,
):
    """Generate R-SMILES text while retaining one-to-one source provenance."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if stable_ids is None:
        stable_ids = [str(i) for i in range(len(reactants))]
    if row_indices is None:
        row_indices = list(range(len(reactants)))
    if not (len(reactants) == len(products) == len(stable_ids) == len(row_indices)):
        raise ValueError("reactants, products, stable_ids, and row_indices must have equal length")
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError(f"Stable IDs are not unique in split {set_name}")

    data = [{
        "reactant": reactant,
        "product": product,
        "augmentation": augmentation,
        "root_aligned": root_aligned,
        "stable_id": str(stable_id),
        "row_index": int(row_index),
        "seed": int(seed) + int(row_index),
    } for reactant, product, stable_id, row_index in zip(
        reactants, products, stable_ids, row_indices
    )]
    src_data = []
    tgt_data = []
    output_index = []
    failures = []
    skip_dict = {
        'invalid_p': 0,
        'invalid_r': 0,
        'error_mapping': 0,
        'error_mapping_p': 0,
        'empty_p': 0,
        'empty_r': 0,
    }
    processes = multiprocessing.cpu_count() if processes < 0 else processes
    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(func=multi_process, iterable=data)
    pool.close()
    pool.join()
    edit_distances = []
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict.setdefault(result['status'], 0)
            skip_dict[result['status']] += 1
            failures.append({
                "stable_id": result["stable_id"],
                "row_index": result["row_index"],
                "status": result["status"],
                "error": result.get("error"),
            })
            continue
        if character:
            for i in range(len(result['src_data'])):
                result['src_data'][i] = " ".join([char for char in "".join(result['src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['tgt_data'][i] = " ".join([char for char in "".join(result['tgt_data'][i].split())])
        edit_distances.append(result['edit_distance'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
        for augmentation_index in range(len(result['src_data'])):
            output_index.append({
                "output_index": len(output_index),
                "row_index": result["row_index"],
                "augmentation_index": augmentation_index,
                "stable_id": result["stable_id"],
            })
    print("Avg. edit distance:", np.mean(edit_distances) if edit_distances else float("nan"))
    print('size', len(src_data))
    for key, value in skip_dict.items():
        print(f"{key}:{value},{value/len(reactants)}")
    if augmentation != 999:
        src_path = os.path.join(save_dir, 'src-{}.txt'.format(set_name))
        tgt_path = os.path.join(save_dir, 'tgt-{}.txt'.format(set_name))
        id_path = os.path.join(save_dir, 'id-{}.txt'.format(set_name))
        index_path = os.path.join(save_dir, 'index-{}.jsonl'.format(set_name))
        failure_path = os.path.join(save_dir, 'failures-{}.jsonl'.format(set_name))
        manifest_path = os.path.join(save_dir, 'manifest-{}.json'.format(set_name))

        with open(src_path, 'w') as f:
            for src in src_data:
                f.write('{}\n'.format(src))
        with open(tgt_path, 'w') as f:
            for tgt in tgt_data:
                f.write('{}\n'.format(tgt))
        with open(id_path, 'w') as f:
            for item in output_index:
                f.write('{}\n'.format(item['stable_id']))
        with open(index_path, 'w') as f:
            for item in output_index:
                f.write(json.dumps(item, sort_keys=True) + '\n')
        with open(failure_path, 'w') as f:
            for item in failures:
                f.write(json.dumps(item, sort_keys=True) + '\n')

        expected_output_rows = len(data) * augmentation
        successful_stable_ids = [
            result["stable_id"] for result in results if result["status"] == 0
        ]
        coverage_ok = (
            not failures
            and len(src_data) == expected_output_rows
            and len(tgt_data) == expected_output_rows
            and len(output_index) == expected_output_rows
            and set(successful_stable_ids) == set(stable_ids)
        )
        manifest = {
            "augmentation": augmentation,
            "coverage_ok": coverage_ok,
            "failed_input_rows": len(failures),
            "input_rows": len(data),
            "input_stable_ids_sha256": _stable_ids_sha256(stable_ids),
            "output_rows": len(src_data),
            "output_stable_ids_sha256": _stable_ids_sha256(
                [item["stable_id"] for item in output_index]
            ),
            "processor": os.path.relpath(__file__, os.getcwd()),
            "processor_sha256": _file_sha256(__file__),
            "root_aligned": root_aligned,
            "seed": seed,
            "split": set_name,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write('\n')

        if strict_coverage and not coverage_ok:
            raise RuntimeError(
                f"Strict coverage failed for {set_name}: "
                f"{len(failures)} failed input rows and "
                f"{len(src_data)}/{expected_output_rows} output rows"
            )
    return src_data, tgt_data


def multi_process(data):
    try:
        return _multi_process(data)
    except Exception as error:
        return {
            "status": "processing_error",
            "src_data": [],
            "tgt_data": [],
            "edit_distance": 0,
            "stable_id": data["stable_id"],
            "row_index": data["row_index"],
            "error": repr(error),
        }


def _multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status":0,
        "src_data":[],
        "tgt_data":[],
        "edit_distance":0,
        "stable_id":data["stable_id"],
        "row_index":data["row_index"],
    }
    # if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(rids)) != len(rids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(pids)) != len(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if pro_mol is not None and not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        rng = random.Random(data["seed"])
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(rng.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(rng.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                # Complete endpoint datasets may retain precursor fragments that do
                # not contribute a mapped atom to the product. Preserve those
                # fragments instead of silently changing the reaction target.
                for i, rea in enumerate(reactant):
                    if i not in used_indices:
                        aligned_reactants.append(clear_map_canonical_smiles(rea))
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([
                clear_map_canonical_smiles(rea) for rea in reactant
            ])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation-1)):
                pro_smi = Chem.MolToSmiles(pro_mol,doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol,doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))

        # Check one representative augmentation per source row. Rooting and
        # fragment ordering may change the strings, but must not change either
        # molecular graph.
        generated_product = "".join(return_status['src_data'][0].split())
        generated_reactant = "".join(return_status['tgt_data'][0].split())
        if clear_map_canonical_smiles(generated_product) != clear_map_canonical_smiles(product):
            return_status["status"] = "product_mismatch"
        elif clear_map_canonical_smiles(generated_reactant) != clear_map_canonical_smiles(".".join(reactant)):
            return_status["status"] = "reactant_mismatch"
        if return_status["status"] != 0:
            return_status['src_data'] = []
            return_status['tgt_data'] = []
            return return_status

        edit_distances = []
        for src,tgt in zip(return_status['src_data'],return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(),tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        type=str,
                        default='USPTO_50K')
    parser.add_argument("-augmentation",type=int,default=1)
    parser.add_argument("-seed",type=int,default=33)
    parser.add_argument("-processes",type=int,default=-1)
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-character", action="store_true")
    parser.add_argument("-canonical", action="store_true")
    parser.add_argument("-postfix",type=str,default="")
    parser.add_argument(
        "-strict_coverage",
        action="store_true",
        help="fail unless every input stable ID produces exactly augmentation outputs",
    )
    args = parser.parse_args()
    print('preprocessing dataset {}...'.format(args.dataset))
    datadir = './dataset/{}'.format(args.dataset)
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"Dataset directory does not exist: {datadir}")
    mechet_datasets = {
        "flower_full",
        "flower_full_audit100",
        "mech_uspto_31k_full",
        "mech_uspto_31k_full_audit100",
    }
    strict_coverage = args.strict_coverage or args.dataset in mechet_datasets
    print(args)
    if args.test_only:
        datasets = ['test']
    elif args.train_only:
        datasets = ['train']
    elif args.test_except:
        datasets = ['val', 'train']
    elif args.validastrain:
        datasets = ['test', 'val', 'train']
    else:
        datasets = ['test', 'val', 'train']

    random.seed(args.seed)
    if args.dataset == "USPTO-MIT":
        datadir = './dataset/{}'.format(args.dataset)
        savedir = './dataset/{}_PtoR_aug{}'.format(args.dataset,args.augmentation)
        savedir += args.postfix
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for i, data_set in enumerate(datasets):
            with open(os.path.join(datadir,f"{data_set}.txt"),"r") as f:
                reaction_list = f.readlines()
                if args.validastrain and data_set == "train":
                    with open(os.path.join(datadir, f"valid.txt"), "r") as f:
                        reaction_list += f.readlines()

                reactant_smarts_list = list(
                    map(lambda x: x.split('>>')[0], reaction_list))
                product_smarts_list = list(
                    map(lambda x: x.split('>>')[1], reaction_list))
                product_smarts_list = list(
                    map(lambda x: x.split(' ')[0], product_smarts_list))
                save_dir = os.path.join(savedir, data_set)

                multiple_product_indices = [i for i in range(len(product_smarts_list)) if "." in product_smarts_list[i]]
                for index in multiple_product_indices:
                    products = product_smarts_list[index].split(".")
                    for product in products:
                        reactant_smarts_list.append(reactant_smarts_list[index])
                        product_smarts_list.append(product)
                for index in multiple_product_indices[::-1]:
                    del reactant_smarts_list[index]
                    del product_smarts_list[index]

                src_data, tgt_data = preprocess(
                    save_dir,
                    reactant_smarts_list,
                    product_smarts_list,
                    data_set,
                    args.augmentation,
                    reaction_types=None,
                    root_aligned=not args.canonical,
                    character=args.character,
                    processes=args.processes,
                    seed=args.seed,
                    strict_coverage=strict_coverage,
                )

    else:
        savedir = './dataset/{}_PtoR_aug{}'.format(args.dataset, args.augmentation)

        savedir += args.postfix
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for i, data_set in enumerate(datasets):
            csv_path = f"{datadir}/raw_{data_set}.csv"
            csv = pd.read_csv(csv_path, low_memory=False)
            reaction_list = list(csv["reactants>reagents>production"])
            if "stable_id" in csv:
                stable_ids = [str(value) for value in csv["stable_id"]]
            elif "id" in csv:
                stable_ids = [str(value) for value in csv["id"]]
            else:
                stable_ids = [f"{args.dataset}:{data_set}:{i}" for i in range(len(csv))]
            row_indices = list(range(len(csv)))
            if args.validastrain and data_set == "train":
                csv_path = f"{datadir}/raw_val.csv"
                csv = pd.read_csv(csv_path, low_memory=False)
                reaction_list += list(csv["reactants>reagents>production"])
                offset = len(row_indices)
                if "stable_id" in csv:
                    stable_ids += [str(value) for value in csv["stable_id"]]
                elif "id" in csv:
                    stable_ids += [str(value) for value in csv["id"]]
                else:
                    stable_ids += [f"{args.dataset}:val:{i}" for i in range(len(csv))]
                row_indices += list(range(offset, offset + len(csv)))

            # random.shuffle(reaction_list)
            reactant_smarts_list = list(
                map(lambda x: x.split('>')[0], reaction_list))
            reactant_smarts_list = list(
                map(lambda x: x.split(' ')[0], reactant_smarts_list))
            reagent_smarts_list = list(
                map(lambda x: x.split('>')[1], reaction_list))
            product_smarts_list = list(
                map(lambda x: x.split('>')[2], reaction_list))
            product_smarts_list = list(
                map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
            print("Total Data Size", len(reaction_list))

            # reaction_class_list = list(map(lambda x: int(x) - 1, csv['class']))
            sub_react_list = reactant_smarts_list
            sub_prod_list = product_smarts_list
            sub_stable_ids = stable_ids.copy()
            sub_row_indices = row_indices.copy()
            save_dir = os.path.join(savedir, data_set)

            # duplicate multiple product reactions into multiple ones with one product each
            multiple_product_indices = [i for i in range(len(sub_prod_list)) if "." in sub_prod_list[i]]
            for index in multiple_product_indices:
                products = sub_prod_list[index].split(".")
                for product_index, product in enumerate(products):
                    sub_react_list.append(sub_react_list[index])
                    sub_prod_list.append(product)
                    sub_stable_ids.append(
                        f"{sub_stable_ids[index]}::product:{product_index}"
                    )
                    sub_row_indices.append(sub_row_indices[index])
            for index in multiple_product_indices[::-1]:
                del sub_react_list[index]
                del sub_prod_list[index]
                del sub_stable_ids[index]
                del sub_row_indices[index]
            src_data, tgt_data = preprocess(
                save_dir,
                sub_react_list,
                sub_prod_list,
                data_set,
                args.augmentation,
                reaction_types=None,
                root_aligned=not args.canonical,
                character=args.character,
                processes=args.processes,
                stable_ids=sub_stable_ids,
                row_indices=sub_row_indices,
                seed=args.seed,
                strict_coverage=strict_coverage,
            )
