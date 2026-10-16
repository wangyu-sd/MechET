#!/usr/bin/env python3
"""Draw only saved committed route states/programs as self-contained SVG/PDF."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from mechet.natural_language_electron_flow import compile_event_arguments
from mechet.frozen_route_assets import file_sha256

FONT=Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
NS='http://www.w3.org/2000/svg'


def electron_points(mol,maps,coords,move):
    def point(endpoint):
        positions=[coords[atom] for atom in endpoint['atoms']]
        x=sum(p[0] for p in positions)/len(positions);y=sum(p[1] for p in positions)/len(positions)
        return (x+9,y-11) if endpoint['kind']=='LP' else (x,y)
    source=move['source'];sink=move['sink'];end=point(sink)
    if sink['kind']=='BOND':
        left,right=sink['atoms']
        if mol.GetBondBetweenAtoms(maps[left],maps[right]) is None:
            other=set(sink['atoms'])-set(source['atoms'])
            if len(other)==1:end=coords[next(iter(other))]
    return point(source),end


def route_panels(graph,report):
    if graph['case_id']!=report['id']:raise ValueError('saved case identity mismatch')
    if not report.get('stock_solved') or not report.get('complete_reaction_edges'):
        raise ValueError('a saved nontrivial stock-connected route is required')
    known={edge['id']:edge for edge in graph['reaction_edges']};panels=[]
    for reaction,edge in enumerate(report['complete_reaction_edges'],1):
        if known.get(edge['id'])!=edge:raise ValueError('reaction differs from saved graph')
        program=edge['program'];actions=program['actions']
        if not actions or actions[-1]['name']!='finish_trace':raise ValueError('missing saved finish')
        if any(not a.get('state_before') or a.get('result',{}).get('ok') is not True for a in actions):
            raise ValueError('missing saved authoritative states/results')
        panels.append({'reaction':reaction,'kind':'target','state':actions[0]['state_before'],'moves':[]})
        for index,action in enumerate(actions[:-1]):
            name=action['name'];state=action['state_before'];moves=[]
            if name=='apply_electron_flow':moves=compile_event_arguments(state,action['arguments'])
            elif name=='import_fragments':state=actions[index+1]['state_before']
            else:raise ValueError('unknown saved tool action')
            panels.append({'reaction':reaction,'kind':'flow' if moves else 'import','state':state,'moves':moves})
        panels.append({'reaction':reaction,'kind':'reaction_endpoint',
                       'state':program['full_terminal_state'],'moves':[]})
    return panels


def molecule_svg(state,moves,uid,*,font=FONT,width=500,height=270):
    params=Chem.SmilesParserParams();params.removeHs=False
    mol=Chem.MolFromSmiles(state,params)
    if mol is None:raise ValueError('saved molecule cannot be drawn')
    rdDepictor.Compute2DCoords(mol)
    maps={a.GetAtomMapNum():a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    active=set();pairs=set();arrows=[]
    for move in moves:
        if move.get('mode')=='BE_DELTA':
            for delta in move.get('bond_deltas',[]):
                active.update(delta['atoms']);pairs.add(tuple(sorted(delta['atoms'])))
            active.update(charge['atom_map'] for charge in move.get('charge_actions',[]))
        else:
            arrows.append(move)
            for endpoint in [move['source'],move['sink']]:
                active.update(endpoint['atoms'])
                if endpoint['kind']=='BOND':pairs.add(tuple(sorted(endpoint['atoms'])))
    if active-set(maps):raise ValueError('saved arrow references absent atoms')
    bonds=[]
    for left,right in pairs:
        bond=mol.GetBondBetweenAtoms(maps[left],maps[right])
        if bond is not None:bonds.append(bond.GetIdx())
    display=Chem.Mol(mol)
    for atom in display.GetAtoms():atom.SetAtomMapNum(0)
    drawer=rdMolDraw2D.MolDraw2DSVG(width,height);options=drawer.drawOptions()
    options.bondLineWidth=2.5;options.padding=.09;options.minFontSize=16;options.maxFontSize=24
    if font and Path(font).exists():options.fontFile=str(font)
    tint=(.87,.91,.86);atoms=[maps[a] for a in active]
    drawer.DrawMolecule(display,highlightAtoms=atoms,highlightBonds=bonds,
        highlightAtomColors={a:tint for a in atoms},highlightBondColors={b:tint for b in bonds})
    coords={n:(drawer.GetDrawCoords(i).x,drawer.GetDrawCoords(i).y) for n,i in maps.items()}
    drawer.FinishDrawing();drawing=ET.fromstring(drawer.GetDrawingText())
    content=''.join(ET.tostring(child,encoding='unicode') for child in drawing)
    content+=f'<defs><marker id="arrow-{uid}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L0,7 L7,3.5z" fill="#667A70"/></marker></defs>'
    for move in arrows:
        start,end=electron_points(mol,maps,coords,move)
        if start is None or end is None:raise ValueError('saved arrow cannot be grounded')
        sx,sy=start;ex,ey=end;dx,dy=ex-sx,ey-sy;distance=max(math.hypot(dx,dy),1)
        bend=min(38,max(18,distance*.2));cx=(sx+ex)/2-dy/distance*bend;cy=(sy+ey)/2+dx/distance*bend
        content+=f'<path data-electron-flow="{move.get("electrons",2)}" d="M{sx},{sy} Q{cx},{cy} {ex},{ey}" fill="none" stroke="#667A70" stroke-width="2.5" marker-end="url(#arrow-{uid})"/>'
    return f'<svg data-molecule="true" xmlns="{NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{content}</svg>'


def render_case_svg(graph,report,*,font=FONT):
    panels=route_panels(graph,report);stock=Counter(report['stock_leaves'])
    columns=3;width=510;row_height=320;start=60+math.ceil(len(panels)/columns)*row_height
    height=start+math.ceil(len(stock)/columns)*row_height
    content=f'<svg xmlns="{NS}" width="{width*columns}" height="{height}" viewBox="0 0 {width*columns} {height}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:"Times New Roman",serif;font-size:24px;fill:black}}</style><text x="18" y="36">Generated route</text>'
    names={'target':'Product','import':'Import','flow':'Inverse electron flow','reaction_endpoint':'Reaction endpoint'}
    for index,panel in enumerate(panels):
        x=(index%columns)*width;y=60+(index//columns)*row_height
        label='Recorded BE Δ' if any(m.get('mode')=='BE_DELTA' for m in panel['moves']) else names[panel['kind']]
        content+=f'<g transform="translate({x},{y})"><text x="18" y="25">{panel["reaction"]}. {label}</text><g transform="translate(0,35)">{molecule_svg(panel["state"],panel["moves"],str(index),font=font)}</g></g>'
    for index,(leaf,count) in enumerate(sorted(stock.items())):
        x=(index%columns)*width;y=start+(index//columns)*row_height
        label='Stock'+(f' × {count}' if count>1 else '')
        content+=f'<g data-stock-leaf="true" transform="translate({x},{y})"><text x="18" y="25">{label}</text><g transform="translate(0,35)">{molecule_svg(leaf,[],f"stock{index}",font=font)}</g></g>'
    return content+'</svg>'


def write_svg_pdf(svg,output,*,conversion_python):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    path=output.with_suffix('.svg');path.write_text(svg)
    # The existing base environment contains CairoSVG; keep model/RDKit
    # environments independent instead of installing into their dependencies.
    subprocess.run([str(conversion_python),'-m','cairosvg',str(path),'-o',
                    str(output.with_suffix('.pdf'))],check=True,capture_output=True)


def load_saved_case(run,mode,case_id):
    from scripts.infer_dynamic_electron_trees import ADAPTER_SHA
    selected=[]
    for complete in (Path(run)/mode).glob('rank*/complete.json'):
        saved=json.loads(complete.read_text());manifest=saved['manifest']
        if (manifest.get('adapter_sha')!=ADAPTER_SHA or manifest.get('full_denominator')!=460
                or manifest.get('evaluation_unit')!='stock_connected_AND_route'
                or manifest.get('route_cohort')!='all' or manifest.get('mode')!=mode):
            raise ValueError('not a frozen full-route experiment shard')
        selected.extend((complete,row,manifest) for row in saved['rows'] if row['id']==case_id)
    if len(selected)!=1:raise ValueError('one saved full-run case required')
    return selected[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case-id',required=True)
    p.add_argument('--mode',choices=['greedy','tree'],default='tree');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font-file',type=Path,default=FONT);args=p.parse_args()
    from scripts.infer_dynamic_electron_trees import ADAPTER_SHA
    complete,report,manifest=load_saved_case(args.run,args.mode,args.case_id)
    graph_path=complete.parent/report['tree_artifact']
    svg=render_case_svg(json.loads(graph_path.read_text()),report,font=args.font_file)
    write_svg_pdf(svg,args.output,conversion_python='/root/miniconda3/bin/python')
    provenance={'case_id':args.case_id,'model_sha256':ADAPTER_SHA,'inference_manifest':manifest,
        'source_complete_sha256':file_sha256(complete),'source_graph_sha256':file_sha256(graph_path),
        'route_key':report['route_key'],'saved_native_states_only':True,'guessed_intermediates':False,
        'chemical_plausibility':report['chemical_plausibility'],
        'font_file_used':str(args.font_file) if args.font_file.exists() else 'RDKit default',
        'svg_sha256':file_sha256(args.output.with_suffix('.svg'))}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print({'phase':'actual-route-figure','case':args.case_id,'output':str(args.output)},flush=True)


if __name__=='__main__':main()
