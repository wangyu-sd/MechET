"""Population panels derived from complete, source-bound LGRV result tables.

Missing reports remain pending. Contexts, signatures, reactions and targets
have separate units; a large unseen-program count is not chemical novelty.
"""
import copy

ROUTE_OUTCOMES=('StockSolved@B','AnyExactStepPrecedent@B',
                'AllStepExactPrecedentSolved@B','PrecedentCoveredSolved@B')
RUNS=('r1_mcts_pr88','r1_greedy_dynamic','r1_greedy_shared')
LABELS={'r1_mcts_pr88':'MCTS','r1_greedy_dynamic':'Greedy I','r1_greedy_shared':'Greedy II'}
COHORT_LABELS={'ursa_expert100':'URSA Expert','ursa_drugs100':'URSA Drugs',
               'retrocast_mkt_cnv160':'RetroCast','paroutes_derived_n5_100':'PaRoutes'}


def prepare_panels(metrics,target_cohorts):
    if metrics.get('ranking_is_primary') is not False:
        raise ValueError('population chemistry panels cannot use reference ranking as primary')
    if metrics.get('matched_compute_claim') is not False:
        raise ValueError('archived-run figures require explicit unequal-work interpretation')
    allowed=set(target_cohorts);cohorts={c:{} for c in dict.fromkeys(target_cohorts.values())}
    totals={};composition={}
    for run in RUNS:
        record=metrics['runs'][run];rows=record['targets']
        if (record['denominator']!=len(allowed) or len(rows)!=len(allowed) or
            {r['target_id'] for r in rows}!=allowed):
            raise ValueError('figure population differs from original frozen target universe')
        for row in rows:
            if row['report_status'] not in {'observed','pending'}:
                raise ValueError('unknown report status')
            if any(not isinstance(row[name],bool) for name in ROUTE_OUTCOMES):
                raise ValueError('route outcomes must preserve boolean target units')
            if row['report_status']=='pending' and any(row[name] for name in ROUTE_OUTCOMES):
                raise ValueError('pending target cannot have a recorded route witness')
            stock,any_exact,all_exact,covered=(row[n] for n in ROUTE_OUTCOMES)
            if ((any_exact or covered) and not stock) or (all_exact and not (any_exact and covered)):
                raise ValueError('route evidence violates stock/all-edge implication')
        def summary(subset):
            return {'denominator':len(subset),'observed_reports':sum(r['report_status']=='observed' for r in subset),
                'pending_reports':sum(r['report_status']=='pending' for r in subset),
                **{name:sum(r[name] for r in subset) for name in ROUTE_OUTCOMES}}
        totals[run]=summary(rows)
        if any(totals[run][k]!=record[k] for k in ['observed_reports','pending_reports']):
            raise ValueError('reported figure missingness disagrees with target rows')
        if any(totals[run][k]!=record['counts'][k] for k in ROUTE_OUTCOMES):
            raise ValueError('reported population counts disagree with target outcomes')
        for cohort in cohorts:
            cohorts[cohort][run]=summary([r for r in rows if target_cohorts[r['target_id']]==cohort])
        counts=metrics.get('composition',{}).get(run)
        if counts is None or counts['denominator']!=len(allowed):
            raise ValueError('authenticated composition census for original targets required')
        composition[run]=copy.deepcopy(counts)
    return {'route_cohorts':cohorts,'route_totals':totals,'composition':composition,
            'tier_counts':dict(metrics['literature_tiers']),'matched_compute_claim':False,
            'interpretation':'recorded-run descriptive; unequal work; pending is not failure',
            'program_chemical_novelty_claim':False,'reference_C0_C1_C2':'Unknown',
            'r2_join_status':metrics.get('r2_complete_program_join','unknown')}


def render_population(data,case,style):
    from scripts.render_nmi_tree_figures import molecule_svg,embed,label,svg_document
    width=style['format']['full_width_mm']*72/25.4
    colors={'r1_mcts_pr88':style['palette']['search_blue_dark'],
            'r1_greedy_dynamic':style['palette']['neutral_dark'],
            'r1_greedy_shared':style['palette']['neutral_mid']}
    def panel(letter,title,y):
        return label(7,y,letter,size=8,weight='bold')+label(20,y,title,size=7,weight='bold')
    def mol(smiles,moves,x,y,w,h,uid):
        svg,_=molecule_svg(smiles,moves,style,uid=uid,width=w,height=h)
        return embed(svg,x,y,w,h)
    def number(value):return 'NA' if value is None else f'{value:,}'
    body=panel('a','Frozen-stock AND closure on original target populations',13)
    for i,run in enumerate(RUNS):
        x=7+i*110
        body+=f'<rect x="{x}" y="24" width="9" height="5" fill="{colors[run]}"/>'+label(x+13,29,LABELS[run])
    body+=label(7,42,'Recorded-run lower bounds; unequal work; unfinished searches remain pending.',size=6)
    for i,(cohort,values) in enumerate(data['route_cohorts'].items()):
        x=7+i*(width-14)/len(data['route_cohorts']);w=(width-14)/len(data['route_cohorts'])
        body+=label(x,58,COHORT_LABELS.get(cohort,cohort),weight='bold')
        for j,run in enumerate(RUNS):
            r=values[run];y=73+j*18;span=w-45
            bar=span*r['StockSolved@B']/r['denominator'] if r['denominator'] else 0
            body+=f'<path d="M{x} {y+1} H{x+span}" stroke="{style["palette"]["neutral_light"]}" stroke-width="5"/>'
            body+=f'<path data-run="{run}" data-cohort="{cohort}" data-stock-count="{r["StockSolved@B"]}" d="M{x} {y+1} H{x+bar}" stroke="{colors[run]}" stroke-width="5"/>'
            body+=label(x+span+5,y+3,f'{r["StockSolved@B"]}/{r["denominator"]}',size=6)
        body+=label(x,133,f'MCTS pending: {values[RUNS[0]]["pending_reports"]}',size=6)
    body+=panel('b','Experimental-reaction evidence along complete stock routes (targets)',154)
    columns=[7,139,227,324,419]
    for x,text in zip(columns,['Policy','Stock AND','Any exact step','Every step exact','Every step A/B']):
        body+=label(x,170,text,size=6,weight='bold')
    for i,run in enumerate(RUNS):
        y=190+i*18;record=data['route_totals'][run]
        body+=label(columns[0],y,LABELS[run])
        for x,key in zip(columns[1:],ROUTE_OUTCOMES):body+=label(x,y,number(record[key]))
    n=sum(data['tier_counts'].values())
    body+=label(7,245,f'All-run reaction evidence: A={data["tier_counts"].get("A",0)}, B={data["tier_counts"].get("B",0)}, C={data["tier_counts"].get("C",0)}, D={data["tier_counts"].get("D",0)}; n={n:,} distinct reactions.',size=6)
    body+=label(7,257,'C denotes unresolved evidence, not chemical invalidity. Evidence is for whole reactions, not arrows.',size=6)
    body+=panel('c','Actual generated net transformations; source scope retained',278)
    # These are selected complete chemical reactions, not a synthetic timeline.
    for i,step in enumerate(case['steps'][:2]):
        y=292+i*83;reaction=step['reaction'];parts=reaction['structural_components']
        citation=step['citation'];status=(citation.get('patent_id') or citation['source_id']) if citation else 'unresolved'
        body+=label(7,y,f'{i+1}. Tier {step["evidence"]["tier"]} · {status}',size=6)
        left=width*.49;cw=left/len(parts)
        for j,smiles in enumerate(parts):
            body+=mol(smiles,[],7+j*cw,y+5,cw-8,64,f'fig2-r{i}-{j}')
            if j<len(parts)-1:body+=label((j+1)*cw-2,y+37,'+',size=7)
        ax=width*.50
        body+=f'<path d="M{ax} {y+35} h24 l-4 -3 m4 3 l-4 3" stroke="{style["palette"]["ink"]}" fill="none"/>'
        body+=mol(reaction['product_raw'],[],width*.57,y+5,width*.41,64,f'fig2-p{i}')
    body+=label(7,462,'Net schemes: forward orientation. Symbolic inverse programs and conditions: linked case plates.',size=6)
    figure2=svg_document(body,style,height_pt=474)

    body=panel('a','Program composition under authenticated actor training exposure',13)
    body+=label(7,28,'Recorded runs; unequal work. Program occurrences, distinct signatures and reactions are different units.',size=6)
    cols=[7,137,221,312,408]
    for x,text in zip(cols,['Policy','Seen contexts','Unseen contexts','Unseen signatures','Reactions with']):
        body+=label(x,46,text,size=6,weight='bold')
    body+=label(cols[-1],54,'unseen programs',size=6,weight='bold')
    for i,run in enumerate(RUNS):
        c=data['composition'][run];y=66+i*20
        values=[LABELS[run],number(c['seen_ordinary']['program_occurrences']),
                number(c['unseen_ordinary']['program_occurrences']),number(c['unseen_ordinary']['unique_program_signatures']),
                number(c['unseen_ordinary']['unique_complete_reactions'])]
        for x,text in zip(cols,values):body+=label(x,y,text)
    body+=panel('b','Whole-reaction precedent for seen versus unseen MCTS programs',133)
    for x,text in zip([7,185,304,413],['Composition','Distinct reactions','A/B-supported reactions','Supported targets']):
        body+=label(x,151,text,size=6,weight='bold')
    for i,kind in enumerate(['seen_ordinary','unseen_ordinary']):
        c=data['composition'][RUNS[0]][kind];y=171+i*20
        values=['Seen' if i==0 else 'Training-unseen',number(c['unique_complete_reactions']),
                number(c['source_supported_unique_reactions']),number(c['source_supported_targets'])]
        for x,text in zip([7,185,304,413],values):body+=label(x,y,text)
    body+=label(7,209,'A reaction may have both seen and unseen programs; reaction counts are not additive. Reference C0/C1/C2: Unknown.',size=6)
    body+=panel('c','Selected recorded inverse FLOW actions, not inferred physical intermediates',232)
    # Prefer an actual unseen program, otherwise a labelled ordinary case. The
    # case spec already fixes the program; do not select by reference rank.
    selected=next((s for s in case['steps'] if s['program']['exposure_class']=='unseen_ordinary'),case['steps'][0])
    cells=[c for c in selected['cells'] if c['kind']=='apply_electron_flow']
    shown=cells if len(cells)<=2 else [cells[0],cells[-1]]
    for i,cell in enumerate(shown):
        x=7+i*(width-14)/max(1,len(shown));w=(width-14)/max(1,len(shown))-8
        body+=label(x,248,f'Action {cell["action_index"]+1} · {len(cell["moves"])} recorded pair moves',size=6)
        body+=mol(cell['state'],cell['moves'],x,255,w,141,f'fig3-flow-{i}')
    body+=label(7,413,'Selected complete program: '+selected['program']['exposure_class'].replace('_',' ')+'. All actions retained in linked plates.',size=6)
    body+=label(7,426,'Source-backed net chemistry does not establish the exact microscopic electron-arrow order.',size=6)
    body+=label(7,442,'Training-unseen electronic composition is not new chemistry, strict-H2 generalization or a reference C1 label.',size=6)
    figure3=svg_document(body,style,height_pt=454)
    return {'fig2_r1_literature_supported_routes':figure2,'fig3_r2_supported_composition':figure3}
