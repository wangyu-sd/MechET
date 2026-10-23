"""Reconcile every saved program and produce a compact offline analysis report."""
import csv
import gzip
import hashlib
from html import escape
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE/'results'
SOURCE = HERE.parent/'nmi_literature_evidence/source_checked_20261019_v18'
NAMES = {
    'proton_transfer':'质子转移',
    'carbonyl_addition':'羰基亲核加成',
    'carbonyl_elimination':'羰基重建／四面体中间体消除',
    'concerted_carbon_substitution':'碳中心直接置换（普通 SN2 适用性检查）',
    'heteroatom_addition_elimination_or_substitution':'磷／硫中心加成、消除或置换',
    'other_multiple_bond_addition_elimination':'其他多重键加成／消除',
    'E2_like_elimination':'E2 型消除',
    'lone_pair_capture_of_cation':'孤对捕获阳离子',
    'isolated_lone_pair_bond_formation':'孤立孤对成键',
    'isolated_bond_heterolysis_or_polarization':'孤立键异裂／极化',
    'bond_migration_or_pi_shift':'键迁移／π电子移动',
    'unclassified_coupled_event':'尚未覆盖的耦合事件',
    'cancelled_arrows_no_transformation':'箭头抵消、无净变化',
    'unavailable':'当前规则不支持的事件／标注未完成',
    'BE_DELTA_without_explicit_arrows':'无显式箭头的 BE_DELTA',
}
STATUS = {'template_compatible':'局部模板兼容', 'conditions_insufficient':'条件或表示信息不足',
          'template_conflict':'特定模板不适用／空操作', 'uncovered':'规则库未覆盖'}


def rows(path):
    with gzip.open(path, 'rt') as f:
        for l in f:
            yield json.loads(l)


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def table(headers, values):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in r)+\
        '</tr>' for r in values)+'</tbody></table>'


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    events, occurrences, errors = {}, Counter(), Counter()
    annotations = OUT/'event_annotations_v2.jsonl.gz'
    if not annotations.exists():
        annotations = OUT/'event_annotations.jsonl.gz'
    for r in rows(annotations):
        occurrences[r['event_id']] += r['program_occurrences']
        if r['event_id'] in events:
            old = events[r['event_id']]
            for key in ('template', 'status', 'state_after', 'inverse_moves'):
                assert old.get(key) == r.get(key), ('same exact event has conflicting annotations', r['event_id'], key)
            old['reaction_keys'] = sorted(set(old['reaction_keys']) | set(r['reaction_keys']))
        else:
            events[r['event_id']] = r
            if 'annotation_error' in r:
                errors[r['annotation_error']] += 1
    source = {r['program_record_id']:r for r in rows(SOURCE/'program_reaction_links.jsonl.gz')}
    links = list(rows(OUT/'program_event_links.jsonl.gz'))
    assert len(links) == len(source) == len({r['program_record_id'] for r in links})
    assert {r['program_record_id'] for r in links} == set(source)
    expected_flow = sum(r['complexity']['flow_events'] for r in source.values())
    observed_flow = sum(len(r['event_ids']) for r in links)
    assert expected_flow == observed_flow == sum(occurrences.values())
    assert len(events) == summary['unique_exact_state_arrow_events']
    status_counts = Counter(r['status'] for r in events.values())
    template_counts = Counter(r['template'] for r in events.values())
    checks = Counter(r.get('transition_check','unavailable') for r in events.values())
    program_stats = defaultdict(Counter)
    byrun = defaultdict(Counter)
    for r in events.values():
        byrun[r['run_id']][r['status']] += 1
    source_evidence = {r['reaction_key']:r for r in rows(SOURCE/'reaction_evidence.jsonl.gz')}
    source_program_status = defaultdict(Counter)
    for link in links:
        e = [events[eid] for eid in link['event_ids']]
        assert len(e) == source[link['program_record_id']]['complexity']['flow_events']
        stats = program_stats[link['run_id']]
        stats['programs'] += 1
        if e and all(x['status'] == 'template_compatible' for x in e):
            stats['all_events_locally_compatible'] += 1
        if any(x['status'] == 'template_conflict' for x in e):
            stats['has_specific_template_conflict'] += 1
        if any(x['status'] == 'uncovered' for x in e):
            stats['has_uncovered_event'] += 1
        tier = source_evidence.get(link['reaction_key'],{}).get('tier','C')
        source_program_status[tier]['programs'] += 1
        source_program_status[tier]['all_events_locally_compatible'] += bool(e) and all(
            x['status'] == 'template_compatible' for x in e)
    totals = {'unique_exact_state_arrow_events':len(events), 'flow_event_program_occurrences':observed_flow,
              'program_records':len(links), 'status_counts':dict(status_counts),
              'template_counts':dict(template_counts), 'transition_checks':dict(checks),
              'annotation_errors':dict(errors), 'by_run_status':dict(byrun),
              'program_coverage_by_run':dict(program_stats),
              'net_reaction_evidence_strata_programs':dict(source_program_status),
              'all_program_ids_and_flow_counts_reconciled':True, 'primary_event_annotations':annotations.name,
              'no_chemical_accuracy_or_endpoint_accuracy_claim':True}
    dump('reconciled_metrics.json', totals)
    with (OUT/'template_counts.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['template','name_zh','unique_exact_events','fraction',*STATUS])
        for k,n in template_counts.most_common():
            c = Counter(r['status'] for r in events.values() if r['template'] == k)
            w.writerow([k,NAMES.get(k,k),n,n/len(events),*(c[s] for s in STATUS)])
    # Keep explicit forward/inverse moves and raw states in each example.
    chosen = {}
    for r in events.values():
        key = (r['template'],r['status'])
        if key not in chosen:
            chosen[key] = r
    preferred_ids = ['ef95080c435b0197438f736786cded868274938f8c43ad99b7478c46b6435cb9',
                     'c49020ce06068cd39246c156c4e405781b2e6a1810637e6afaaeff86a316e5ee']
    cab = []
    for pid in preferred_ids:
        link = next(r for r in links if r['program_record_id'] == pid)
        cab.append({'program_record_id':pid, 'reaction_product':source[pid]['reaction_product'],
                    'mapped_structural_precursors':source[pid]['mapped_structural_precursors'],
                    'net_reaction_evidence':source_evidence[link['reaction_key']],
                    'events':[events[eid] for eid in link['event_ids']]})
    dump('cabozantinib_complete_event_analysis.json', cab)
    dump('case_gallery.json', list(chosen.values()))
    # Selected HTML cases use compact active-centre labels, with original mapped
    # states and full argument payloads available in expandable raw details.
    sections = []
    for program in cab:
        cells=[]
        for r in program['events']:
            arrows=[]
            for m in r.get('inverse_moves',[]):
                def c(v):
                    return v['kind']+'('+','.join(map(str,v['atoms']))+')'
                arrows.append(c(m['source'])+' → '+c(m['sink']))
            cells.append([r['action_index']+1,NAMES.get(r['template'],r['template']),
                          STATUS[r['status']],'; '.join(arrows),'; '.join(r.get('flags',[]))])
        sections.append('<h3>'+escape(program['reaction_product'])+'</h3>'+table(
            ['原始动作序号','对应的正向模板','局部检查','实际逆向箭头（私有映射号）','未决事项'],cells)+
            '<details><summary>完整输入、输出和动作</summary><pre>'+escape(json.dumps(program,ensure_ascii=False,indent=2))+'</pre></details>')
    status_table = table(['结论','事件数','比例'],[[STATUS[k],status_counts[k],f'{100*status_counts[k]/len(events):.2f}%'] for k in STATUS])
    template_table = table(['正向对应模板／事件族','事件数','比例'],[
        [NAMES.get(k,k),n,f'{100*n/len(events):.2f}%'] for k,n in template_counts.most_common()])
    run_table=table(['冻结推理条件','程序数','全部事件局部兼容','至少一项特定模板不适用','至少一项未覆盖'],[
        [k,v['programs'],v['all_events_locally_compatible'],v['has_specific_template_conflict'],v['has_uncovered_event']]
        for k,v in sorted(program_stats.items())])
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>逆向电子事件模板分析</title>
<style>body{font:16px/1.65 Georgia,"Times New Roman",serif;max-width:1160px;margin:36px auto;padding:0 20px;color:#171717}h1{font-size:27px}h2{font-size:21px;margin-top:30px}h3{font-size:16px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border-bottom:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}th{border-top:2px solid #222}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#203f65}</style>
<h1>保存的逆向电子事件：模板依据与适用范围</h1>'''
    html+=f'<p>分析对象：冻结 PR89 v18 中全部 {len(links):,} 条完整程序、{observed_flow:,} 次 FLOW 事件出现。按推理条件、动作前状态和原始参数精确去重后为 {len(events):,} 项；不是去对称后的独立化学事件数。未重新调用模型。</p>'
    html+='<p>先将整组逆向箭头反转为正向对应事件，再检查参与原子、键级、电荷和离去基。所有原始树及动作载荷均与冻结哈希核对。标签是人工编写的局部图规则，不是机理真值分类器。</p>'
    html+='<h2>结果</h2>'+status_table+template_table
    html+='<p>“局部模板兼容”仅指所检查的局部条件吻合，仍未验证能垒、pKa、溶剂、竞争反应或中间体寿命。“特定模板不适用”主要指普通 SN2 条件不满足，不排除 SNAr、催化偶联等其他机制。未覆盖不等于化学错误。程序表后三列存在重叠，不应相加。</p>'
    html+='<h2>按冻结推理条件划分</h2>'+run_table
    html+='<h2>卡博替尼：两段完整程序的逐事件检查</h2>'+''.join(sections)
    html+='<h2>教材依据</h2><p>以下资料支持一般机理模式，不证明本次生成路径的微观次序。未将教材正文或图像复制进训练数据。</p><ul>'+''.join(
        '<li><a href="'+escape(u)+'">'+escape(k)+': '+escape(u)+'</a></li>' for k,u in summary['references'].items())+'</ul>'
    html+='<h2>可复查文件</h2><ul>'+''.join('<li><a href="'+p+'">'+p+'</a></li>' for p in [
        'reconciled_metrics.json','template_counts.csv','cabozantinib_complete_event_analysis.json','case_gallery.json',
        annotations.name,'program_event_links.jsonl.gz'])+'</ul></html>'
    (OUT/'report.html').write_text(html)
    manifest={'source_program_file':str(SOURCE/'program_reaction_links.jsonl.gz'),
              'source_program_sha256':summary['source_sha256'],
              'scripts':{},'outputs':{}}
    for p in HERE.glob('*.py'):
        manifest['scripts'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in OUT.iterdir():
        if p.is_file() and p.name!='delivery_manifest.json':
            manifest['outputs'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump('delivery_manifest.json',manifest)
    print(json.dumps(totals,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
