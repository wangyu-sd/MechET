"""Source-bound chemical cases; net precedent and symbolic arrows stay separate."""
from collections import defaultdict
import html
import math
import textwrap

from .literature_evidence import assess
from .literature_programs import bind_program_states
from .literature_reaction_keys import canonical_components, make_reaction, stable_hash
from .literature_route_metrics import route_evidence,source_supported


def _pointer(value, pointer):
    for key in pointer.lstrip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def render_net_precedent_gallery(queries,primary_sources,style,*,reaction_keys):
    """Selected actual complete reactions; never a population success estimate."""
    from scripts.render_nmi_tree_figures import molecule_svg,embed,label,svg_document
    by_key={r['reaction_key']:r for r in queries}
    sources={s['source_id']:s for s in primary_sources}
    if not reaction_keys or len(set(reaction_keys))!=len(reaction_keys):
        raise ValueError('gallery requires distinct selected frozen reactions')
    height=53+107*len(reaction_keys)
    if height*25.4/72>style['format']['max_height_mm']:
        raise ValueError('split gallery across pages, never silently omit selected cases')
    width=style['format']['full_width_mm']*72/25.4
    body=label(7,13,'Performed experimental precedent for generated complete reactions',size=7,weight='bold')
    body+=label(7,25,'Selected cases; complete net chemistry, not experimentally established electron-arrow sequences.',size=6)
    metadata={'reaction_keys':list(reaction_keys),'physical_mechanism_verified':False,
              'selection':'explicit source-positive gallery, not population validation','cases':[]}
    for i,key in enumerate(reaction_keys):
        if key not in by_key:raise ValueError('gallery reaction absent from frozen generated pool')
        reaction=by_key[key]
        if make_reaction(reaction['product_raw'],reaction['structural_components'])['reaction_key']!=key:
            raise ValueError('gallery reaction payload differs from frozen identity')
        evidence=assess(reaction,[],primary_sources)
        if not source_supported(evidence):raise ValueError('gallery case lacks checked performed reaction precedent')
        source=sources[evidence['selected_source']];y=40+i*107
        body+=label(7,y,f'{i+1}. Tier {evidence["tier"]} · {source["source_id"]}',size=6.5,weight='bold')
        parts=reaction['structural_components'];left=width*.49;cell_width=left/len(parts)
        for j,smiles in enumerate(parts):
            drawing,_=molecule_svg(smiles,[],style,uid=f'gallery-{i}-r{j}',width=cell_width-9,height=61)
            body+=embed(drawing,7+j*cell_width,y+6,cell_width-9,61)
            if j<len(parts)-1:body+=label((j+1)*cell_width-1,y+38,'+',size=7)
        x=width*.51
        body+=f'<path d="M{x} {y+36} h24 l-4 -3 m4 3 l-4 3" stroke="{style["palette"]["ink"]}" fill="none"/>'
        drawing,_=molecule_svg(reaction['product_raw'],[],style,uid=f'gallery-{i}-p',width=width*.41,height=61)
        body+=embed(drawing,width*.58,y+6,width*.41,61)
        conditions=source.get('source_conditions','Source conditions unavailable.')
        for j,line in enumerate(textwrap.wrap(conditions,120)):
            if j>=4:raise ValueError('source conditions require separate larger case plate')
            body+=label(7,y+78+j*7,line,size=5.5)
        metadata['cases'].append({'reaction_key':key,'evidence':evidence,
            'source_id':source['source_id'],'source_url':source['source_url'],'locator':source['locator'],
            'source_conditions':conditions,'source_document_sha256':source['document_sha256'],
            'generated_occurrence_source':reaction.get('source'),
            'raw_product':reaction['product_raw'],'structural_components':parts})
    body+=label(7,height-6,'Source/training overlap unknown; an experimentally documented transformation is not a tested whole route.',size=6)
    return svg_document(body,style,height_pt=height),metadata


def prepare_case(route, occurrences, programs, tree, primary_sources):
    """Resolve selected native actions and recompute both reaction and evidence.

    The caller authenticates file hashes. This boundary also checks native
    terminal ownership, payload hashes, root/endpoint continuity and structural
    reaction identity so a stale signature cannot draw an unrelated program.
    No arrows are derived from a reaction's net bond difference.
    """
    from mechet.endpoints import split_precursor_endpoints
    from mechet.natural_language_electron_flow import compile_event_arguments
    if tree['case_id']!=route['target_id']:
        raise ValueError('case native tree belongs to another target')
    by_reaction=defaultdict(list)
    for occurrence in occurrences:
        if (occurrence['run_id'],occurrence['target_id'])!=(route['run_id'],route['target_id']):
            raise ValueError('case occurrences mix another run or target')
        rebuilt=make_reaction(occurrence['product_raw'],occurrence['structural_components'])
        if rebuilt['reaction_key']!=occurrence['reaction_key']:
            raise ValueError('case reaction payload differs from frozen key')
        by_reaction[occurrence['reaction_key']].append(occurrence)
    chosen={}
    for program in programs:
        if ((program['run_id'],program['target_id'])!=(route['run_id'],route['target_id']) or
            program['reaction_key'] in chosen):
            raise ValueError('one selected program per route reaction and target required')
        chosen[program['reaction_key']]=program
    if set(chosen)!=set(route['reaction_keys']):
        raise ValueError('selected programs do not cover every essential route reaction')
    sources={s['source_id']:s for s in primary_sources}
    steps=[];evidence={}
    for key in route['reaction_keys']:
        if key not in by_reaction:
            raise ValueError('route edge missing frozen reaction occurrence')
        occurrence=by_reaction[key][0];program=chosen[key]
        pointer=program['program_source_pointer']
        if not pointer.endswith('/actions') or '/contexts/' not in pointer:
            raise ValueError('case program is not a native terminal action pointer')
        context=_pointer(tree,pointer.rsplit('/',1)[0])
        inner_pointer=pointer.rsplit('/contexts/',1)[0]
        inner=_pointer(tree,inner_pointer) if inner_pointer else tree
        contexts={c['id']:c for c in inner['contexts']}
        if len(contexts)!=len(inner['contexts']):
            raise ValueError('duplicate native context IDs')
        root=contexts[inner['root_context']]['state']
        actions=context['actions']
        if (context.get('terminal') is not True or context['id']!=program['native_context_id'] or
            stable_hash(actions)!=program['program_payload_sha256']):
            raise ValueError('native terminal or program payload differs')
        binding=bind_program_states(actions,root,context['state'])
        if not binding['valid']:
            raise ValueError('invalid case program state binding: '+binding['reason'])
        if canonical_components(root)!=canonical_components(occurrence['product_raw']):
            raise ValueError('native program root differs from frozen product')
        parts=split_precursor_endpoints(context['state'],root)
        actual=make_reaction(root,parts.structural)
        if actual['reaction_key']!=key:
            raise ValueError('native terminal differs from frozen structural reaction')
        cells=[]
        for index,action in enumerate(actions):
            kind=action['name'];cell={'action_index':index,'kind':kind,'moves':[]}
            if action.get('result',{}).get('ok') is not True:
                raise ValueError('case action has no accepted native result')
            if kind=='import_fragments':
                fragments=action['arguments']['fragments']
                if not fragments or any(f.get('count',1)<1 for f in fragments):
                    raise ValueError('empty or invalid import payload')
                cell.update(fragments=fragments)
            elif kind=='apply_electron_flow':
                if action['arguments'].get('direction')!='retrosynthetic':
                    raise ValueError('not a recorded inverse electron-flow action')
                cell.update(state=action['state_before'],
                            moves=compile_event_arguments(action['state_before'],action['arguments']))
            elif kind=='finish_trace':
                cell['state']='.'.join(actual['structural_components'])
                cell['auxiliary_components']=list(parts.auxiliary)
            else:
                raise ValueError('unrecognized native action')
            cells.append(cell)
        evidence[key]=assess(occurrence,[],primary_sources)
        citation=sources.get(evidence[key]['selected_source'])
        steps.append({'reaction_key':key,'reaction':occurrence,'program':program,'cells':cells,
                      'state_binding':binding,'evidence':evidence[key], 'citation':citation})
    return {**route_evidence(route,evidence),'steps':steps,'physical_mechanism_verified':False}


def render_case(case, style, *, title):
    """One net-chemistry plate plus complete, paginated inverse action plates."""
    from scripts.render_nmi_tree_figures import molecule_svg, embed, label, svg_document
    width=style['format']['full_width_mm']*72/25.4
    ink=style['palette']['ink'];teal=style['palette']['terminal_teal_dark']
    def molecule(state,moves,x,y,w,h,uid):
        drawing,info=molecule_svg(state,moves,style,uid=uid,width=w,height=h)
        return embed(drawing,x,y,w,h),info
    def lines(x,y,text,*,chars=100,size=6.5):
        return ''.join(label(x,y+i*(size+2),s,size=size) for i,s in enumerate(textwrap.wrap(text,chars)))
    # Each reaction has its own row. Counts do not imply an equal-work rescue.
    height=47+len(case['steps'])*150
    if height*25.4/72>style['format']['max_height_mm']:
        raise ValueError('case net scheme exceeds publication page; select a separate multi-page case')
    body=label(7,13,title,size=8,weight='bold')
    closure='All leaves in frozen stock' if case['nonzero_stock_solved'] else 'No nonzero stock-closure witness'
    body+=label(7,26,closure+' · '+str(len(case['steps']))+' essential net reactions')
    metadata={'route_key':case['route_key'],'target_id':case['target_id'],
              'precedent_covered':case['precedent_covered'],'steps':[],
              'physical_mechanism_verified':False,'net_chemistry_orientation':'forward'}
    details=[]
    for step_index,step in enumerate(case['steps']):
        y=38+step_index*150;reaction=step['reaction'];evidence=step['evidence']
        citation=step['citation'];components=reaction['structural_components']
        status=('Tier '+evidence['tier']+' · '+(citation.get('patent_id') or citation['source_id'])) if citation else 'Unresolved · no checked primary source'
        body+=label(7,y,str(step_index+1)+'. '+status,weight='bold')
        # Separate reactant drawings prevent a long disconnected SMILES from
        # shrinking every molecule to the scale of the largest component.
        left_width=width*.47;cell_width=left_width/len(components)
        for i,component in enumerate(components):
            drawing,_=molecule(component,[],7+i*cell_width,y+9,cell_width-9,78,f'net-{step_index}-r{i}')
            body+=drawing
            if i<len(components)-1:body+=label(7+(i+1)*cell_width-8,y+49,'+',size=8)
        x=width*.49
        body+=f'<path d="M{x} {y+45} H{x+32} l-5 -3 m5 3 l-5 3" stroke="{ink}" stroke-width="1" fill="none"/>'
        drawing,_=molecule(reaction['product_raw'],[],width*.57,y+9,width*.41-7,78,f'net-{step_index}-p')
        body+=drawing
        if citation:
            body+=lines(7,y+103,citation.get('source_conditions','Source conditions not recorded'),chars=115)
            if citation.get('experimental_product_status'):
                body+=lines(7,y+130,citation['experimental_product_status'],chars=120,size=6)
        else:
            body+=label(7,y+103,'No source support is inferred from symbolic execution.')
        action_meta=[]
        for offset in range(0,len(step['cells']),6):
            chunk=step['cells'][offset:offset+6];rows=math.ceil(len(chunk)/2)
            page_height=42+rows*141
            detail=label(7,13,f'{title} · inverse program {step_index+1} · actions {offset+1}–{offset+len(chunk)}',size=7,weight='bold')
            detail+=label(7,26,'Recorded IMPORT / FLOW / FINISH order; executor states, not observed intermediates.')
            for local,cell in enumerate(chunk):
                x=(local%2)*width/2;ycell=36+(local//2)*141;w=width/2-12
                i=cell['action_index'];kind=cell['kind'];molecules=[]
                if kind=='import_fragments':
                    detail+=label(x+7,ycell+4,f'{i+1}. IMPORT · added species')
                    fragments=cell['fragments'];fw=w/len(fragments)
                    for j,fragment in enumerate(fragments):
                        drawing,info=molecule(fragment['smiles'],[],x+7+j*fw,ycell+15,fw-3,98,f'import-{step_index}-{i}-{j}')
                        detail+=f'<g data-import-fragment="{html.escape(fragment["smiles"],quote=True)}" data-count="{fragment.get("count",1)}">'+drawing+label(x+9+j*fw,ycell+123,'×'+str(fragment.get('count',1)))+'</g>'
                        molecules.append(info)
                else:
                    name='FLOW · '+str(len(cell['moves']))+' electron-pair moves' if kind=='apply_electron_flow' else 'FINISH · structural precursors'
                    detail+=label(x+7,ycell+4,f'{i+1}. '+name)
                    drawing,info=molecule(cell['state'],cell['moves'],x+7,ycell+11,w,116,f'event-{step_index}-{i}')
                    detail+=drawing;molecules.append(info)
                action_meta.append({'action_index':i,'kind':kind,'page_index':len(details),'molecules':molecules})
            details.append(svg_document(detail,style,height_pt=page_height))
        metadata['steps'].append({'reaction_key':step['reaction_key'],'program':step['program'],
            'evidence':evidence,'action_indices':[c['action_index'] for c in step['cells']],
            'action_drawings':action_meta})
    footer=('Primary sources support net transformations; exact symbolic arrows are displayed separately.'
            if all(source_supported(s['evidence']) for s in case['steps']) else
            'Reaction evidence remains unresolved or conflicting; symbolic execution is not experimental support.')
    body+=label(7,height-5,footer,size=6)
    return svg_document(body,style,height_pt=height),details,metadata
