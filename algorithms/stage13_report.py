"""Generate a provenance-labelled paper readout; never changes policy or outcomes."""
import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
DEFAULT_ROOT = LAB / 'outputs/three_regrasp_20261122_01'


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def dist(values):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p90': None}
    at = .9 * (len(values) - 1)
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p90': values[lo] * (1 - at + lo) + values[hi] * (at - lo)}


def describe_distribution(value):
    return '/'.join(number(value.get(k)) for k in ('mean', 'median', 'p90'))


def wilson(k,n=40):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f"{100*(center-half):.1f}–{100*(center+half):.1f}%"


def policy_timing(row):
    """Policy-side wall outside backend requests, including initial planning.

    This is wall time, including scheduler/IPC idle; it is not process CPU time.
    Native backend monotonic stamps avoid mixing unrelated elapsed clocks.
    """
    path = Path(row.get('episode', '')) / 'wall_profile_backend.json'
    profile = load(path)
    marks = profile.get('marks', {})
    begin, end = marks.get('ready'), marks.get('finish_request')
    if begin is None or end is None or end < begin:
        return {'scope': 'backend full-span timing unavailable'}
    intervals = sorted((max(begin, r['start_wall_s']), min(end, r['end_wall_s']))
                       for r in profile.get('active_request_intervals', [])
                       if r.get('end_wall_s', begin) > begin and r.get('start_wall_s', end) < end)
    merged = []
    for a, b in intervals:
        if b < a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    service = sum(b-a for a,b in merged)
    rpc=load(Path(row.get('episode',''))/'rpc_timing.json')
    first_action_start=min((r['start_wall_s'] for r in rpc if r.get('op')=='step'),default=end) if isinstance(rpc,list) else end
    initial_service = sum(max(0., min(first_action_start,b)-a) for a,b in merged if a < first_action_start)
    return {'policy_span_s': end-begin, 'backend_rpc_service_s': service,
            'policy_nonrpc_wall_s': max(0.,end-begin-service),
            'initial_policy_nonrpc_s': max(0.,first_action_start-begin-initial_service),
            'recorded_between_dispatch_compute_s': row.get('policy_compute_s'),
            'scope': 'ready-to-finish monotonic wall minus union of backend active request intervals; includes initial planning, policy scheduling and IPC idle; not process CPU time',
            'backend_profile_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def trial_label(row):
    if not row:
        return '未运行'
    sequence = '→'.join(str(c['contact_count']) for c in row.get('closures', [])) or '0'
    outcome = '成功' if row.get('success') else row.get('failure_class', '失败')
    return f"{outcome}; {row.get('actual_grasps', 0)}抓; {sequence}指"


def history_evidence(plan, root, freeze_dir):
    """Count historical mentions for this exact object list, never fixed totals."""
    ids = [row['object_id'] for row in plan.get('objects', [])]
    for path in (freeze_dir / 'history_exposure.json', root / 'audit/history_exposure.json'):
        data = load(path)
        if 'history_exposure' in data:
            data = data['history_exposure']
        if set(ids) <= set(data.get('objects', {})):
            return dict(data, objects={key: data['objects'][key] for key in ids},
                        source_audit=str(path))
    environment = load(root / 'audit/environment_data_audit.json')
    manifest = plan.get('source_manifest') or environment.get('data', {}).get('source_manifest')
    source_lab = Path(manifest).parent / 'FingerEyeLab' if manifest else None
    if source_lab is not None and (source_lab / 'experiments').is_dir():
        from stage13_dataset_audit import history_audit
        return dict(history_audit(ids, source_lab), source_lab=str(source_lab),
                    input_scope='current frozen object IDs; original source experiments only')
    return dict(scope='Historical source experiment audit unavailable; no unseen-object claim.',
                objects={key: dict(exposed=None) for key in ids}, scanned_files=None)


def cohort_status(plan, frozen, summary, detail):
    """Only exact registered names, objects, conditions and source can be pooled."""
    expected = {row['name']: row for row in plan.get('runs', [])}
    source = frozen.get('source_commit')
    if not summary and not detail:
        return False, None
    valid = bool(source and summary.get('episodes') == len(detail) and
                 set(summary.get('source_commits', [])) == {source} and
                 set(detail) <= set(expected))
    for name, row in detail.items():
        registered = expected.get(name, {})
        valid = valid and all(row.get(key) == registered.get(key) for key in ('name','object_id','condition')) and row.get('source_commit') == source
    if not valid:
        return False, '现存逐例汇总与所选冻结的提交、命令名、物体或条件不一致，暂不汇入正式统计；旧队列仅保留为历史协议预跑。'
    complete = bool(summary.get('complete') and len(expected) == len(detail) == 80 and
                    set(detail) == set(expected) and len({r['object_id'] for r in expected.values()}) == 40 and
                    len(summary.get('pairs', [])) == 40 and
                    all(summary.get('conditions', {}).get(c, {}).get('n') == 40 for c in ('clean','noise')))
    return complete, None


def audit_matches(data, key, detail, source, root):
    records = data.get(key, [])
    if not data or not isinstance(records, list) or len(records) != len(detail) or {r.get('name') for r in records} != set(detail):
        return False
    for row in records:
        commit = row.get('source_commit')
        compact = root/'formal_runs'/row['name']/'compact.json'
        digest = row.get('compact_sha256')
        if digest and (not compact.is_file() or hashlib.sha256(compact.read_bytes()).hexdigest() != digest):
            return False
        # Contact audit binds its parent compact instead of duplicating source_commit.
        if commit != source and not (commit is None and digest and load(compact).get('source_commit') == source):
            return False
    return True


def protocol_history_lines(root):
    specs=[('protocol_amendment.json','source_commit','protocol_pilot_dc15ee8','两条件初始俯视RGB未逐像素共用'),
           ('protocol_geometry_amendment.json','old_frozen_commit','protocol_geometry_pilot_117244a','同一解析圆柱的126/127侧面离散被重复计为两个物体'),
           ('protocol_diversity_amendment.json','old_commit','protocol_diversity_pilot_19bc1d5','精确去重仍保留近球、接近比例圆柱及近立方体，未达到预先要求的形状差异')]
    rows=[]
    for filename,key,directory,reason in specs:
        data=load(root/'audit'/filename)
        archive=data.get('archived_cohort') or directory
        actual=Path(archive);actual=actual if actual.is_absolute() else root/actual
        count=data.get('archived_receipt_count',len(list(actual.glob('*/batch_result.json'))))
        rows.append([data.get(key,'审计未登记'),reason,count,str(archive),data.get('whole_cohort_invalidated',data.get('old_cohort_scope','整队排除，逐例结局不影响排除'))])
    return ['## 三轮协议预跑与策略版本', '',
            table(['预跑提交','整队作废原因','归档回执','归档位置','排除范围'],rows),'',
            '三轮已启动队列全部排除，不挑选其余对象或有利结局拼入最终80回合。前两轮分别更正初始RGB配对与跨细分几何身份；第三轮进一步冻结几何家族差异规则。物体替换依据几何与零抓取资格，不依据抓取成败；原不利预跑对象`abc_53930`按登记优先级保留。',
            '原始修订分别见[RGB协议](audit/protocol_amendment.json)、[精确几何去重](audit/protocol_geometry_amendment.json)与[形状差异协议](audit/protocol_diversity_amendment.json)。这些审计中的“当时策略/守卫未改”只描述对应预跑修订，不能外推到V4候选。','']


def diversity_lines(root, expected_ids):
    proposal=load(root/'audit/diversity_replacement_proposal.json')
    order=proposal.get('selected_candidate_order',[])
    candidates={row['object_id']:row for row in proposal.get('candidates',[])}
    rows=[]
    for key in order:
        row=candidates.get(key,{})
        rows.append([key,key in expected_ids,number(row.get('uniform_scale'),10),
                     ' × '.join(number(v*1000,3) for v in row.get('scaled_dimensions_m',[])),
                     row.get('native_orientation_patch',{}).get('passed','未登记'),row.get('source_sha256','未登记')])
    return ['## 几何家族5%差异规则与五个候选', '',
            '精确几何身份检查继续忽略平移、旋转和统一缩放；额外的家族差异规则用于排除几乎相同的普通形状。圆柱以解析长径比例比较，差异为两比例绝对差除以较大值；长方体比较排序并归一化的三尺寸，差异为各维相对差的最大值。差异≤5%连边，形成的连通分量只保留最早预登记顺序的一个；新候选与保留对象须超过该门槛。同组近球还须满足拟合轴比≤1.01及连续径向壳相对厚度≤1%，尺寸缩放不增加新身份。',
            '这是一项明确的几何多样性代理，并非任意曲面的数学非同构证明；无法由基元解析的形状仍依赖登记的表面/距离谱核验。传递分组可比单对5%更保守。五个替代候选按几何来源、变换和指垫区域预检登记；下表的原生姿态静态预检不能替代实际稳定摆放资格，也不能算作物理抓取验收。', '',
            table(['候选ID','已在所选冻结清单','统一缩放','登记尺寸mm','静态指垫预检','源网格SHA256'],rows),'',
            f"当前提案顺序：`{order}`；提案排除的近重复ID：`{proposal.get('removed_ids',[])}`。最终是否选入，以所选冻结的40对象计划及零抓取资格哈希为准；未冻结或未资格通过的候选不得自动变成正式分母。",
            '逐对公式、分组与阈值见[几何差异分析](audit/diversity_v3_analysis.json)，五候选来源与预检见[候选提案](audit/diversity_replacement_proposal.json)。','']


def guard_acceptance_lines(root, expected_commit):
    names=('cube_two','cube_three','cube_auto','pyramid_auto3')
    directory=root/'guard_acceptance_readout';rows=[];missing=[]
    for name in names:
        row=load(directory/(name+'.json'))
        if not row:
            missing.append(name)
            rows.append([name,'未运行','—','—','—','—','—','—'])
            continue
        sequence='→'.join(str(c['contact_count']) for c in row.get('closures',[])) or '0'
        rows.append([name,row.get('source_commit','未登记'),row.get('success'),row.get('actual_grasps','—'),sequence,
                     sum(c.get('branch')=='preload' for c in row.get('closures',[])),
                     row.get('controller_completed','未登记'),row.get('source_commit')==expected_commit if expected_commit else '尚未冻结'])
    return ['## 新增预载速度守卫的独立物理验收', '',
            '以下仅读取`guard_acceptance_readout/`的新版本记录，全部排除于正式80回合。固定三指若实际0抓仍如实记为规划拒绝，不能把完成一个仿真回合解释为三指闭合成功。缺失记录保留“未运行”，不借用a1bbe71或任一作废队列。', '',
            table(['验收案例','实际源提交','成功','实际抓次','实际指组','其中预载复验','控制器完成','源提交同最终冻结'],rows),'',
            f"尚缺验收记录：`{missing}`。验收提交与最终冻结若不同，需按各自提交及运行时哈希解释，不自动宣称相同策略。真实守卫停止、物理阶段与后续回执在对应逐例JSON及启动记录中核对；没有后续ack不抹去已有motor-phase执行。",'']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--readout', type=Path)
    parser.add_argument('--freeze', type=Path, default=os.environ.get('FINGEREYE_FINAL_FREEZE'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    readout = (args.readout or root / 'formal_readout').resolve()
    output = (args.output or root / 'EVALUATION.md').resolve()
    selected_freeze = args.freeze or Path('frozen_v4')
    freeze_dir = (selected_freeze if selected_freeze.is_absolute() else root/selected_freeze).resolve()
    freeze_label = freeze_dir.name
    frozen = load(freeze_dir / 'freeze.json')
    plan = load(freeze_dir / 'plan.json')
    summary = load(readout / 'SUMMARY.json')
    selection_audit = load(root / 'selection_readout/selection_audit.json')
    contact_audit = load(root / 'contact_readout/contact_audit.json')
    segmented_timing = load(root / 'timing_readout/timing.json')
    detail = {}
    for path in sorted(readout.glob('*.json')):
        row = load(path)
        if 'closures' in row:
            detail[row['name']] = row
    complete, readout_notice = cohort_status(plan, frozen, summary, detail)
    expected_ids = {row['object_id'] for row in plan.get('objects', [])}
    expected_commit = frozen.get('source_commit')
    if readout_notice:
        summary, detail = {}, {}
    supplemental_unavailable=[]
    def current_audit(data,key,label):
        records=data.get(key,[])
        if not data:return {}
        if not audit_matches(data,key,detail,frozen.get('source_commit'),root):
            supplemental_unavailable.append(label)
            return {}
        return data
    selection_audit=current_audit(selection_audit,'episodes','选指')
    contact_audit=current_audit(contact_audit,'episodes_data','接触')
    segmented_timing=current_audit(segmented_timing,'episodes','分阶段时间')
    timing_audit = {name: dict(object_id=row.get('object_id'),condition=row.get('condition'),source_commit=row.get('source_commit'),**policy_timing(row)) for name,row in detail.items()}
    validation=load(root/'audit/final80_validation.json')
    qa_complete=bool(complete and validation.get('final_complete') is True and validation.get('all_available_checks_passed') is True and validation.get('source_commit')==expected_commit and validation.get('controller_sha256')==frozen.get('sha256') and Path(validation.get('freeze_directory','/unavailable')).resolve()==freeze_dir)
    lines = ['# FingerEyeLab 三指与重抓配对评测', '',
             f"状态：{'正式80条回执已齐（含基础设施失败，完整物理证据另列）' if complete else '尚未满足正式80回合完成条件；本报告为进度快照，不作为完整论文结果'}。",
             f"正式结果已返回 {summary.get('episodes', 0)}/80 回合；统计单位为40个不同物体，每个物体两种视觉条件。", '',
             '## 配置与分母', '',
             f"基线提交：`7fa3bd37752a38d5b2dc65da6735804c78057a09`。正式候选提交：`{frozen.get('source_commit', '尚未冻结')}`。",
             f"正式结果记录提交：`{', '.join(summary.get('source_commits', [])) or '尚无正式记录'}`。",
             f"早期三指/恢复实现提交为`a1bbe71ae86f79ba5f5f818352a75dc3eeb50460`；其物理验收只证明该历史版本。当前正式版本取自`{freeze_label}/freeze.json`，对应提交`{frozen.get('source_commit', '尚未冻结')}`，不能把旧版验收移标为V4验收。",
             'V4候选在`stage5_control.py::s5_move_segment`把`s13_preload_recovery`接入已有实测速度检查，保留平移0.15 m/s与转动1.5 rad/s的守卫阈值及原始本体反馈。该变更会改变真实命令的继续/停止分支，属于新策略/动作守卫版本，不是仅改报告或几何登记；需要本版本重新回归和物理冒烟，旧提交的连续三抓轨迹只能作为历史证据。',
             '主配置显式为 **fast + auto + 最多3次实际抓取**。120 Hz物理、20 Hz控制、30 fps普通RGB与录像；普通评测采用minimal留档，仍生成策略需要的六路FingerEye输入。',
             '主评分为`lift30_no_drop_v1`：命令抬升至少29.5 mm，实际抬升至少28 mm，净空至少5 mm，满足末尾连续2 s的保持、无掉落及完整性门槛；名义试抬30 mm。旧严格评分单列。',
             '实际抓次以成功下发并有物理步记录的闭合动作计；规划候选、手型选择或被守卫拒绝的命令不计。原位调预载后的重新试抬占一次物理尝试预算，并与新闭合分开标记。文中的实际两指/三指指真实执行该指组闭合；不自动等同于所有手指都已接触物体，接触证据须另查物理接触记录。', '',
             '干净条件保留原视觉路径；加噪条件只在策略实际读取的六路FingerEye RGB中，以浮点数加入零均值高斯噪声，σ=3/255（8-bit尺度3灰度级），再裁剪回合法范围。相机/曝光帧/预登记种子决定噪声，同一曝光重复读取像素一致；初始俯视图不加该附加噪声。原路径的额外RGB噪声为0，两组共享其余原有传感设置、初态、物理、输入时序及动作守卫。',
             '本轮物理执行证据来自Isaac仿真的连续动力学回合。在线策略仅使用普通RGB、本体及静态标定；仿真物体位姿和接触仅供评测及离线统计。',
             f"两种条件逐物体共用零抓取几何资格回合开始时采集的同一普通初始俯视RGB，使用前校验文件哈希、图像规格及静态相机标定。每回合新渲染俯视RGB另存策略不可读的审计位置。所选冻结输入清单：`{plan.get('initial_overhead_pairing',{}).get('manifest','尚未冻结')}`；初始RGB、几何身份与形状差异修订分别单列。",
             f"返回完整性：已完成控制器 {summary.get('fully_completed_episodes', 0)}/80，完整性有效 {summary.get('integrity_valid_episodes', 0)}/80，基础设施异常 {summary.get('infrastructure_episodes', 0)}；异常仍留在相应40对象分母。完整物理证据标志：{summary.get('complete_physical_evidence', False)}。", '']
    lines += protocol_history_lines(root)
    lines += diversity_lines(root, expected_ids)
    lines += [f"最终原始证据QA完成且匹配所选冻结：{qa_complete}。80回执齐全与逐帧/物理判据核验是分别报告的条件。", '']
    if readout_notice:
        lines += [readout_notice, '']
    if supplemental_unavailable:
        lines += ['尚未与当前逐例集合和冻结提交完全对应的补充审计已暂不汇入：'+ '、'.join(supplemental_unavailable)+'。', '']
    qualification = load(root / 'audit/settled_qualification.json')
    rules = qualification.get('rules', {})
    qualification_rows = {row['object_id']: row for row in qualification.get('objects', [])}
    qualified_ids = expected_ids & set(qualification_rows)
    qualified_passed = sum(qualification_rows[key].get('passed') is True for key in qualified_ids)
    qualification_complete = bool(len(expected_ids) == 40 and qualified_ids == expected_ids and
                                  qualified_passed == len(expected_ids) and (root/'audit/settled_qualification.json').is_file() and hashlib.sha256((root/'audit/settled_qualification.json').read_bytes()).hexdigest()==frozen.get('geometry_qualification_sha256'))
    history = history_evidence(plan, root, freeze_dir)
    history_rows = history.get('objects', {})
    exposed = sum(row.get('exposed') is True for row in history_rows.values())
    no_reference = sum(row.get('exposed') is False for row in history_rows.values())
    history_unknown = len(expected_ids) - exposed - no_reference
    replacement = load(root / 'audit/replacement_geometry_proposal.json')
    geometry_amendment = load(root / 'audit/protocol_geometry_amendment.json')
    proposed = replacement.get('proposed_unqualified_object', {})
    transform = proposed.get('source_transform', {})
    selection_description = plan.get('selection', '尚未登记')
    geometry_description = plan.get('distinct_shape_test', '尚未登记')
    lines += ['## 物体筛选与冻结', '',
              f"当前清单登记{len(expected_ids)}个不同物体、{len(plan.get('runs', []))}回合；与该清单对应的零抓取几何资格已返回{len(qualified_ids)}/{len(expected_ids)}，通过{qualified_passed}个，全部合格：{qualification_complete}。",
              '在正式抓取结果产生前，按稳定摆放几何筛选，禁止按抓取成败替换物体。几何身份忽略文件名、平移、旋转和统一缩放副本。原方块/圆柱与三棱锥只作开发/诊断，不并入正式分母。',
              f"最低稳定高度{number(rules.get('minimum_settled_height_m', .008)*1000)} mm，最高{number(rules.get('maximum_settled_height_m', .05)*1000)} mm；可用指垫支持半径{number(rules.get('pad_support_radius_m', .0012)*1000)} mm、最大切面下陷{number(rules.get('maximum_tangent_sag_m', .0003)*1000)} mm、侧面指垫到桌面净空至少{number(rules.get('minimum_side_footprint_clearance_m', .002)*1000)} mm，并要求相对的普通侧面。",
              f"当前冻结的选入规则：{selection_description}。几何身份核验规则：{geometry_description}。来源调查是按登记的尺寸、面数、侧面和几何身份规则选取样本，不宣称已经穷尽所有可抓形状，也不把不同网格细分视作不同几何。指垫半径与0.3mm切面下陷是冻结的几何代理，未经过柔顺指垫承载标定。",
              f"按当前{len(expected_ids)}个ID重新统计历史实验源文件引用：{exposed}个检出历史引用，{no_reference}个在已扫描范围未检出，{history_unknown}个未知。扫描范围：{history.get('scope', '未提供')}。未检出引用不等于从未用于调参，历史提及也不等于已有物理成功；逐对象引用位置见[历史引用审计](history_exposure.json)。",
              f"冻结文件：`{freeze_dir / 'freeze.json'}`；运行时清单：`{freeze_dir / 'runtime_manifest.json'}`；资格证据：`{root / 'audit/settled_qualification.json'}`。", '']
    if replacement:
        dimensions = ' × '.join(number(value*1000, 3) for value in proposed.get('dims_m', [])) or '未登记'
        replacement_registered = proposed.get('object_id') in expected_ids
        lines += [f"几何修订：旧清单的`{replacement.get('excluded_duplicate', '未登记')}`与保留对象`{replacement.get('same_source_shape_retained', '未登记')}`属于同一圆柱，127/126段网格细分使旧等顶点数距离谱筛查漏检。旧冻结提交为`{geometry_amendment.get('old_frozen_commit', '未登记')}`；整批移作预跑后，登记用`{proposed.get('object_id', '未登记')}`替换rank {proposed.get('selection_rank', '未登记')}，当前清单已包含替换对象：{replacement_registered}。",
                  f"替换按几何规则从原生尺寸未入选候选中确定，使用统一比例{number(transform.get('uniform_scale'), 10)}缩放至最大尺寸30 mm，登记尺寸为{dimensions} mm；统一缩放保留形状比例，不产生额外物体身份。原资产SHA256：`{proposed.get('source_sha256', '未登记')}`；缩放资产SHA256：`{proposed.get('sha256', '未登记')}`。替换在该对象抓取前登记，须先通过零抓取摆放和接触区域资格；选择是否使用抓取成败：{replacement.get('selection_uses_grasp_outcomes', '未知')}。",
                  '替换来源、变换、初态、种子及选择规则见[几何替换提案](audit/replacement_geometry_proposal.json)，整批作废范围见[几何协议修订](audit/protocol_geometry_amendment.json)，跨网格重复的具体见证见[几何交叉核验](audit/geometry_tessellation_crosscheck.json)。', '']
    figure_sources = load(root / 'figures/objects40_sources.json')
    figure_ids = {row.get('object_id') for row in figure_sources.get('sources', [])}
    if (root/'figures/objects40.png').exists() and figure_ids == expected_ids:
        lines += ['![冻结40物体几何目录](figures/objects40.png)', '', '全部预登记物体，按冻结顺序展示稳定摆放姿态；每格独立等比例缩放，h为几何资格检查后的实际高度，不能按画面大小比较实物尺寸。含重复形状家族，几何比例各异；不是40个语义类别。可导出的[PDF](figures/objects40.pdf)与[网格和高度来源](figures/objects40_sources.json)一并保留。', '']
    groups = summary.get('conditions', {})
    if detail:
        lines += ['## 首抓、三抓与视觉噪声', '']
        rows = []
        for cond, title in [('clean', '原视觉'), ('noise', '加噪')]:
            group = groups.get(cond, {})
            rows.append([title, group.get('n', 0), f"{group.get('successes', 0)}/40",
                         f"{100*group.get('successes', 0)/40:.1f}%", group.get('first_successes', 0),
                         group.get('second_added', 0), group.get('third_added', 0),
                         '/'.join(str(group.get('attempt_distribution', {}).get(str(i), 0)) for i in range(4)),
                         number(group.get('actual_grasps', {}).get('mean')),
                         number(group.get('successful_actual_grasps', {}).get('mean'))])
        lines += [table(['条件','已返回','最终成功/固定分母','最终率','首抓成功','第二抓新增','第三抓新增','实际0/1/2/3抓','全部平均抓次','成功平均抓次'], rows), '',
                  '未返回回合不会被解释为失败；进度快照中的固定分母百分比不构成完整成功率估计。80条预登记回执齐全后报告每条件40对象的整体结果；基础设施失败保留为失败并单列，未采集到的初态和图像仍标为未知，不冒充完整物理配对。', '']
        lines += [table(['条件','规划拒绝','物理失败','设施超时','其他设施错误'],
                        [[title] + [groups.get(cond, {}).get('failure_classes', {}).get(k, 0) for k in ('planning_rejection','physical_failure','infrastructure_timeout','infrastructure_error')]
                         for cond,title in [('clean','原视觉'),('noise','加噪')]]), '',
                  f"原视觉成功→加噪失败：{', '.join(summary.get('clean_success_noise_failure', [])) or '无已确认配对'}。",
                  f"原视觉失败→加噪成功：{', '.join(summary.get('clean_failure_noise_success', [])) or '无已确认配对'}。",
                  f"初态不一致对象：{', '.join(summary.get('paired_state_mismatches', [])) or '无已确认不一致'}；初始俯视RGB不一致对象：{', '.join(summary.get('paired_overhead_mismatches', [])) or '无已确认不一致'}。未能采集初始图的对象：{', '.join(summary.get('paired_overhead_unavailable', [])) or '无'}；缺失保留为未知，不记为已核对相同。", '']
        if complete:
            lines += ['对象层面的Wilson95%区间（每条件n=40；不是把80回合当成80个独立对象）：', '', table(['条件','首抓率95%CI','最终率95%CI'], [[title,wilson(groups[cond]['first_successes']),wilson(groups[cond]['successes'])] for cond,title in [('clean','原视觉'),('noise','加噪')]]), '']
            b=len(summary.get('clean_success_noise_failure',[]));c=len(summary.get('clean_failure_noise_success',[]));n=b+c
            exact=min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
            lines += [f'噪声配对不一致计数b={b}, c={c}；双侧exact McNemar p={exact:.4g}。每对象只有一个预登记种子，区间和检验描述本协议下的有限样本，不能证明对任意噪声或初态泛化。','']
        lines += ['## 实际选指和恢复', '']
        for cond,title in [('clean','原视觉'),('noise','加噪')]:
            group=groups.get(cond,{})
            lines += [f"{title}：", '', table(['统计','两指','三指'], [[label, group.get(key,{}).get('2',0),group.get(key,{}).get('3',0)] for key,label in [('first_actual_fingers','首抓实际物体数'),('attempt_fingers','全部实际尝试数'),('final_success_fingers','最终成功实际指组')]]), '',
                      f"auto可行性选择范围：`{json.dumps(group.get('selection_scope', {}), ensure_ascii=False)}`。",
                      f"真实连续转换：`{json.dumps(group.get('transitions', {}), ensure_ascii=False)}`；大幅手势变化{group.get('large_gesture_changes',0)}次，换指组{group.get('hand_switches',0)}次，两者重叠{group.get('large_gesture_and_hand_switch',0)}次，缺失差值证据{group.get('incomplete_gesture_evidence',0)}次。", '']
            recovery = group.get('recovery_counts', {})
            lines += [table(['恢复分支','考虑','选择','动作执行','最后救回','守卫拒绝','有条件后备'], [[branch] + [recovery.get(branch, {}).get(key,0) for key in ('considered','selected','executed','rescued','guard_refused','conditional_fallback_considered')] for branch in ('preload','local_pose','release_resample')]), '']
        if selection_audit:
            lines += ['auto搜索与实际下发的独立审计：','',table(['条件','选择调用','两组均有记录','两组均通过全部检查','实际闭合','缺少下发回执'], [[title] + [selection_audit.get('conditions',{}).get(cond,{}).get(key,0) for key in ('selection_events','both_groups_logged','both_groups_feasible','actual_dispatches','dispatch_receipts_missing')] for cond,title in [('clean','原视觉'),('noise','加噪')]]),'',
                      '局部细化调用可能只搜索当前指组，故`only_one_group_feasible`不能单独证明另一组也被搜索后拒绝。只有`both_groups_logged=true`的调用能支持这一判断；两组均可行时的分数、排序、选中候选与实际下发回执分别在[选择表](selection_readout/selection_events.csv)和[实际下发表](selection_readout/actual_dispatches.csv)中绑定。','']
            gate_rows = []
            for cond,title in [('clean','原视觉'),('noise','加噪')]:
                for count in (2,3):
                    candidates = [r for r in selection_audit.get('candidate_groups',[]) if r.get('condition')==cond and r.get('contact_count')==count]
                    gate_rows.append([title,count]+[sum(r.get(key,0) for r in candidates) for key in ('recorded_candidates','ik_independently_passed','ik_then_endpoint_passed','accepted_path_feasible','path_rejected','path_not_evaluated')])
            lines += [table(['条件','指组','记录候选','IK独立通过','IK后端点通过','完整路径通过','路径拒绝','路径未评估'],gate_rows),'',
                      '该表计数单位为候选/搜索调用，不能作对象成功数。端点诊断可能在更早IK失败后仍被计算；“IK后端点通过”要求前置检查也通过。未到达路径检查不等于路径已失败。逐对象、逐调用原始计数与首个拒绝条件见[分层门控表](selection_readout/candidate_gate_counts.csv)，连杆/间隙/阈值及关节见[完整选择审计](selection_readout/selection_audit.json)。','']
        if contact_audit:
            lines += ['真实接触离线核对：','',table(['条件','实际尝试','尝试内采样到承载手指','末尾2秒承载手指','成功尝试承载手指'],[[title, contact_audit.get('conditions',{}).get(cond,{}).get('attempts',0)]+[json.dumps(contact_audit.get('conditions',{}).get(cond,{}).get(key,{}),ensure_ascii=False) for key in ('loaded_finger_groups','hold_loaded_finger_groups','successful_attempt_loaded_groups')] for cond,title in [('clean','原视觉'),('noise','加噪')]]),'',
                      '承载接触定义为仿真法向力>1e−6 N，按6个物理步即0.05 s采样，仅离线使用。近邻碰撞对和下发指组均不能替代承载证据；未采到接触不能证明采样间隔内绝无接触。每次尝试在真实恢复命令前截断，末尾2 s另列；逐例力、连杆和源文件哈希见[接触审计](contact_readout/contact_audit.json)与[逐次接触CSV](contact_readout/contact_attempts.csv)。','']
        branches = summary.get('branch_counts', {})
        lines += [table(['恢复分支','考虑','选择','动作执行','最后救回','守卫拒绝','有条件后备'], [[branch] + [branches.get(branch, {}).get(key,0) for key in ('considered','selected','executed','rescued','guard_refused','conditional_fallback_considered')] for branch in ('preload','local_pose','release_resample')]), '',
                  '分支选择、恢复动作执行、下一次实际抓取与最终救回是不同事件；“执行释放/退让”本身不表示已经闭合。最后救回由物理评分离线核算，不能用视觉支持判断代替。',
                  '大幅手势改变预先定义为同一预闭合阶段的物体相对腕姿变化≥30°，或共同手指实际指垫平均落点移动≥5 mm。原始弧度/米差值及与换指组的重叠保存在`transitions.csv`；该统计允许离线使用物体位姿。', '']
        lines += ['## 时间（秒）', '', '各单元格为平均/中位数/P90；P90采用相邻次序统计量的线性插值。物理时间按120 Hz实际步数计算，墙钟包含仿真启动、计算、渲染、日志及结束处理。', '']
        timing_names = [('physical_s','机器人动作物理时间'),('first_trial_physical_s','首抓物理阶段'),('retry_physical_s','恢复与再抓额外物理时间'),('policy_compute_s','已记录命令间策略墙钟子集'),('batch_wall_s','端到端墙钟')]
        lines += [table(['时间类型','原视觉 平均/中位/P90','加噪 平均/中位/P90'], [[title] + [describe_distribution(groups.get(cond,{}).get('timing_s',{}).get(key,{})) for cond in ('clean','noise')] for key,title in timing_names]), '',
                  table(['时间样本','原视觉','加噪'],[[label]+[groups.get(cond,{}).get(key,0) for cond in ('clean','noise')] for key,label in [('timing_complete_episodes','完整物理/策略时间'),('batch_wall_timing_complete_episodes','精确端到端墙钟'),('censored_timing_episodes','入口墙钟缺失或中断')]]), '',
                  '启动异常前只记录到的物理时间是该中断回合的已观察值，逐例保留但不冒充完成回合耗时。若恢复的调度回执缺少精确入口墙钟，仅从入口墙钟统计中排除；完整物理步与策略单调时钟仍计入其各自统计。缺失数量以当前逐例标志和上表为准。旧`117244a`预跑的两份回执恢复和串行回退分别记录在[audit/formal80_recovered_receipts.json](audit/formal80_recovered_receipts.json)与[串行回退](audit/formal80_serial_fallback.json)，不并入当前正式统计。', '',
                  '策略计算字段来自审计记录；命令RPC墙钟、渲染和物理执行另计。若记录只覆盖命令间计算，不能将其解释为全部CPU求解时间。精细分段及原始相位统计见逐例JSON的`phase_wall_components`、`phase_physical_s`和`wall_profile`。', '']
        observed_wall={cond:dist(row.get('batch_wall_s') for row in detail.values() if row.get('condition')==cond and row.get('batch_wall_observed_exact')) for cond in ('clean','noise')}
        lines += [table(['全部回执的已观察时间','原视觉 n；平均/中位/P90','加噪 n；平均/中位/P90'], [
            ['已记录物理秒（含设施中断；中断值可能仅为下界）']+[str(groups.get(cond,{}).get('observed_all_physical_s',{}).get('n',0))+'；'+describe_distribution(groups.get(cond,{}).get('observed_all_physical_s',{})) for cond in ('clean','noise')],
            ['可用精确端到端墙钟（含设施失败）']+[str(observed_wall[cond]['n'])+'；'+describe_distribution(observed_wall[cond]) for cond in ('clean','noise')],
            ['设施失败的已观察端到端墙钟']+[str(groups.get(cond,{}).get('observed_infrastructure_batch_wall_s',{}).get('n',0))+'；'+describe_distribution(groups.get(cond,{}).get('observed_infrastructure_batch_wall_s',{})) for cond in ('clean','noise')]]), '',
            '上一主时间表使用完成回合；此表补齐全部回执中可用的实际耗时。缺失精确墙钟的回执保留在成败与抓次分母，时间统计明确列出样本量。', '']
        phase_names=sorted(set(k for group in groups.values() for k in group.get('phase_physical_s',{})))
        if phase_names:
            lines += [table(['实际相位','原视觉物理秒 平均/中位/P90','加噪物理秒 平均/中位/P90'], [[phase] + [describe_distribution(groups.get(cond,{}).get('phase_physical_s',{}).get(phase,{})) for cond in ('clean','noise')] for phase in phase_names]), '']
        if timing_audit:
            timing_groups = {cond:[timing_audit[name] for name,row in detail.items() if row.get('condition')==cond] for cond in ('clean','noise')}
            lines += ['包含首次规划的策略侧墙钟补充（排除全部后端RPC服务区间）：','',table(['范围','原视觉 平均/中位/P90','加噪 平均/中位/P90'],[[label]+[describe_distribution(dist(row.get(key) for row in timing_groups[cond])) for cond in ('clean','noise')] for key,label in [('policy_nonrpc_wall_s','全部策略侧非RPC墙钟'),('initial_policy_nonrpc_s','首次物理动作前策略侧墙钟'),('backend_rpc_service_s','后端RPC服务墙钟')]]),'','补充使用同一后端单调时钟的ready→finish时间跨度减去所有RPC服务区间的并集，包含首次规划、策略进程调度及IPC空闲，不能称为进程纯CPU时间；逐例数值、范围和源文件SHA256见`timing_audit.json`。','']
        if segmented_timing:
            lines += [f"分阶段可加和核对：{segmented_timing.get('valid_count',0)}/{segmented_timing.get('episode_count',0)}回合时间完整。缺失时间仅从时间汇总中排除，绝不移出成败分母。",'']
            stage_names = [('initial_planning','首次规划'),('first_grasp','首抓至首次恢复前'),('release_retreat','释放与退让'),('replanning','恢复后观察与再规划'),('regrasp','恢复动作与再抓')]
            stage_rows = []
            for stage,label in stage_names:
                for metric,metric_label in [('physical_s','物理秒'),('policy_nonrpc_wall_s','策略侧非RPC墙钟'),('wall_s','后端跨度墙钟')]:
                    stage_rows.append([label,metric_label]+[describe_distribution(segmented_timing.get('groups',{}).get(cond,{}).get('stages',{}).get(stage,{}).get(metric,{})) for cond in ('clean','noise')])
            lines += [table(['阶段','量','原视觉 平均/中位/P90','加噪 平均/中位/P90'],stage_rows),'',
                      '首次规划含首次观察及IK；首抓阶段含入场、闭合及试抬验证。首次实际恢复动作后的释放/退让按记录相位分类；之后全部非RPC空隙和纯观察请求归入再规划，其余物理动作归入再抓（含局部动作/预载复验）。五阶段墙钟严格加和为后端ready→finish；另加启动/结束残差才是端到端墙钟。这是墙钟归因而非进程纯CPU时间；并发调度、IPC等待也计入。详见[分段定义和汇总](timing_readout/TIMING.md)、[逐例CSV](timing_readout/timing_stages.csv)和[加和核对JSON](timing_readout/timing.json)。','']
    else:
        lines += ['## 正式成功、抓次、选指、恢复与时间统计', '', '所选冻结版本尚无可汇入的正式逐例结果；本节不填入未运行的成功数、动作数或耗时。历史预跑不补入最终分母。', '']
    lines += ['## 40对象完整配对表', '']
    paired={}
    for row in detail.values():paired.setdefault(row.get('object_id'),{})[row.get('condition')]=row
    all_objects=plan.get('objects',[])
    lines += [table(['对象ID','原视觉：结局/真实抓次/指组序列','加噪：结局/真实抓次/指组序列','初态噪声种子'], [[obj['object_id'],trial_label(paired.get(obj['object_id'],{}).get('clean')),trial_label(paired.get(obj['object_id'],{}).get('noise')),obj.get('case_base',{}).get('noise_seed','—')] for obj in all_objects]), '']
    lines += ['## 历史同初态开发对照（不进正式分母）', '']
    dev_rows=[]
    for dirname,label in [('baseline_readout','基线7fa3bd3'),('development_readout_a1bbe71','历史候选a1bbe71')]:
        for path in sorted((root/dirname).glob('*.json')):
            row=load(path)
            if 'actual_grasps' not in row or not row.get('name'):continue
            dev_rows.append([label,row['name'],row.get('source_commit','—'),str(row.get('success')),row.get('actual_grasps'),'/'.join(str(c['contact_count']) for c in row.get('closures',[])) or '0',number(row.get('physical_s')),number(row.get('batch_wall_s'))])
    lines += [table(['版本','案例','结果提交','成功','实际抓次','实际指组序列','物理秒','墙钟秒'],dev_rows), '',
              '固定三指没有实际闭合的案例属于规划拒绝，不能算作三指物理能力验证。若auto只有两指可行，选择两指不证明两种可行手型之间的优劣。这些物理小样本只属于表中旧提交，不能验收新增实测速度守卫的V4，也不能替代40对象统计。', '']
    lines += guard_acceptance_lines(root, expected_commit)
    auto_rows=[]
    for dirname,label in [('baseline_readout','基线7fa3bd3'),('development_readout_a1bbe71','历史候选a1bbe71')]:
        for name in ('cube_auto','cylinder_auto'):
            row=load(root/dirname/(name+'.json'))
            for event in row.get('selection_events',[]):
                data=event['data']
                auto_rows.append([label,name,data.get('step','—'),data.get('selected','—'),data.get('selection_scope','未记录'),row.get('actual_grasps',0),'→'.join(str(c['contact_count']) for c in row.get('closures',[])) or '未下发'])
    if auto_rows:
        lines += ['自动选指逐例原因：','',table(['版本','案例','决策步','选中指组','可行性范围','实际抓次','下发指组'],auto_rows),'']
    same_load=load(root/'audit/same_load_two_three.json')
    if same_load:
        load_rows=[]
        for count,data in same_load.items():
            nominal=data.get('task',{}).get('nominal',{})
            robust=data.get('task',{}).get('robust',{})
            assumptions=nominal.get('assumptions',{})
            load_rows.append([count,data.get('geometry_feasible'),number(nominal.get('minimum_task_margin')),number(robust.get('minimum_task_margin')),robust.get('task_feasible'),number(nominal.get('lateral_margin')),number(nominal.get('torque_margin')),assumptions.get('mass_range_kg','—'),data.get('motion_scope','—')])
        assumptions=same_load.get('2',{}).get('task',{}).get('nominal',{}).get('assumptions',{})
        lines += ['同负载离线任务比较（假设模型，非实测力）：','',table(['指组','几何可行','名义任务裕量','不确定性任务裕量','不确定性任务可行','名义侧扰裕量','名义力矩裕量','质量假设kg','运动成本范围'],load_rows),'',
                  f"共同假设：质量{assumptions.get('mass_range_kg','—')} kg、摩擦下界{assumptions.get('friction_lower','—')}、每指法向力上限{assumptions.get('max_normal_force_n','—')} N、抬升加速度{assumptions.get('lift_acceleration_m_s2','—')} m/s²、水平扰动{assumptions.get('horizontal_disturbance_n','—')} N、力矩扰动{assumptions.get('torque_disturbance_nm','—')} N·m。",
                  '三指未通过几何时，其理论抗扰裕量不能转为可执行方案优势；两指名义裕量超过1也不代表不确定性任务裕量超过1。质量、摩擦、力限、平面接触足迹和扰动是公开假设，不能称作测得的托重能力。完整模型、运动成本范围和约束见[audit/same_load_two_three.json](audit/same_load_two_three.json)。','']
    rejection=load(root/'audit/baseline_cube_first_rejection.json')
    witness=rejection.get('recomputed_candidate',{}).get('first_rejection_detail',{}).get('geometry',{}).get('first_failed_guard',{})
    candidate=rejection.get('original_candidate',{})
    lines += ['## 可复核的计算链', '',
              '历史三指拒绝案例：基线`cube_three`，提交`7fa3bd3`。`visual_controller.py::_s12_rgb_contact_regions`从公开RGB包络生成区域，`_s12_region_ik`与`Robot.ik`先后求区域/固定落点IK，`_s12_complete_candidate`执行逐指位置、法向、关节限位和端点检查；`SelfAwareGeometry.check`首次拒绝，因此接近/闭合/抬升路径不会下发。候选任务评分调用`stage5_control.py::s12_task_quality`与`general_grasp_contacts.py::friction_wrench_feasibility`，先可行性门控，再按同一负载及扰动假设比较力/力矩裕量、间隙与运动代价。',
              f"原候选逐指位置误差（mm）：`{[round(v*1000,5) for v in candidate.get('per_finger_error_m',[])]}`；法向误差（rad）：`{candidate.get('normal_error_rad',[])}`。",
              f"首个具体失败：`{witness.get('kind','见JSON')}`；连杆`{witness.get('links',[])}`；间隙{number(witness.get('clearance_m',0)*1000,5)} mm，阈值{number(witness.get('threshold_m',0)*1000,5)} mm。机器人自身碰撞和RGB包络碰撞分别保存在`audit/baseline_cube_first_rejection.json`，带原关节、落点、阈值与哈希。",
              '位置门槛为逐指0.8 mm、法向门槛0.5 rad，原候选均通过；桌面最小净空2.99968 mm也通过0.65 mm门槛。其后首先拒绝食指相机component3壳与中指link2的0.10556 mm重叠；另有中指相机component1/component3及link2与RGB包络约6.02615/3.59163/4.57084 mm重叠。自身碰撞与物体包络碰撞为独立检查，不能用其中一项替代另一项。',
              '机器人自身几何碰撞具有独立拒绝证据；即使离线真值表明RGB包络过大，也不能据此忽略自身碰撞。该审计复用记录的关节和公开RGB，用当前等价守卫补充有符号见证；不是一次新的物理执行。', '']
    lines += ['三指搜索把腕部侧倾、食指/中指高度差、指间距与接触面作为区域布局/初值变量；区域IK和固定落点IK都带净空残差，第二次IK后重新检查全部端点与路径。保留的三指方块/圆柱开发结果仍全部拒绝，没有为达到三指执行目标放宽原守卫。每次搜索实际枚举的布局、腕部初值、逐指高度与角向偏移、42 s搜索预算及首个拒绝条件保存在候选JSON；有界搜索失败不能证明全空间无三指解。详细数据、函数和中间命令见[计算链说明](audit/COMPUTATION_CHAINS.md)。','']
    search_rows = []
    for name in ('cube_three','cylinder_three'):
        row=load(root/'development_readout_a1bbe71'/(name+'.json'))
        candidates=[candidate for event in row.get('selection_events',[]) for candidate in event.get('data',{}).get('candidates',[])+event.get('data',{}).get('rejected',[]) if candidate.get('contact_count')==3]
        regions=[region for candidate in candidates for region in candidate.get('contact_regions',[])]
        heights=[region['height_fraction'] for region in regions if 'height_fraction' in region]
        angles=[region['angular_offset_rad'] for region in regions if 'angular_offset_rad' in region]
        q=[candidate['q'] for candidate in candidates if len(candidate.get('q',[]))>=6]
        ranges='; '.join(f"{key}=[{min(v[index] for v in q):.3f},{max(v[index] for v in q):.3f}]" for key,index in [('yaw',3),('pitch',4),('roll',5)]) if q else '未记录'
        search_rows.append([name,len(candidates),len({c.get('layout') for c in candidates if 'layout' in c}),sorted({c.get('wrist_joint_start') for c in candidates if 'wrist_joint_start' in c}),f'{min(heights):.2f}–{max(heights):.2f}' if heights else '—',f'{min(angles):.2f}–{max(angles):.2f}' if angles else '—',ranges])
    lines += [table(['三指开发案例','记录候选','布局数','腕部初值编号','表面高度比例范围','角向偏移范围rad','第二次IK后的腕部ZYX角范围rad'],search_rows),'',
              '初值0沿用公开几何配准；初值1在配准后ZYX角上加[0.50,−0.45,0.35] rad并改变指关节；初值2用三指静态法向与RGB目标法向做刚体配准。表中腕角是实际记录的优化终点范围，不能解释为连续穷举。两个开发案例记录的表面主轴均为[1,0,0]，12布局×3初值共36候选；恢复后可重采主轴，但本表不声称已覆盖所有方向。','']
    development=load(root/'development_readout_a1bbe71/pyramid_auto3.json')
    if development:
        events=[event['data'] for event in development.get('recovery_events',[]) if event['data'].get('event') in ('failure_decision','selected','guard_refused','executed','grasp_dispatched')]
        lines += ['历史实际重抓案例：开发诊断`pyramid_auto3`，提交`a1bbe71`，同一连续物理回合，最多三抓。`stage13_recovery.py::policy_main`由`s13_evaluate`判断公开失败证据，依次调用`s13_preload`及`s13_release_recover`，后者经`visual_controller.py::stage3_recover`尝试`s13_local_probe`。`s13_legal_start`保留原实测q，生成合法命令种子并验证连接；新图证据不足时不强行调预载，局部候选不可行则继续完整释放、退开、重新观察和采样。最终`s5_close`经`s5_step`调用`ctx.step`，并在成功派发后写`grasp_dispatched`。', '',
                  table(['事件','物理步','分支','闭合起始步/判定'], [[e['event'],e.get('step','—'),e.get('branch','—'),e.get('started_step',e.get('reason',e.get('decision',{}).get('reason','—')))] for e in events]), '',
                  f"物理轨迹核对的实际闭合起始步：`{[c['start_step'] for c in development.get('closures',[])]}`。其最终成功为`{development.get('success')}`，因此该案例证明第二/第三次闭合真正下发，不证明重抓救回。实际目标q、反馈q、候选ID与规划哈希在逐例JSON和`agent_trace.jsonl`中绑定；`steps.jsonl`提供120 Hz物理执行记录。", '']
    lines += ['## 旧严格评分附表', '',
              table(['条件','lift30_no_drop_v1成功','旧严格成功'],[[title,groups.get(cond,{}).get('successes','—'),groups.get(cond,{}).get('legacy_strict_successes','—')] for cond,title in [('clean','原视觉'),('noise','加噪')]]), '',
              '## 视频、复跑和证据', '',
              '视频复跑不进入正式分母，需保留原配置、种子和提交，并核对抓次、实际指组与结局。复跑核对清单及4–6段代表视频由`videos/manifest.json`登记；未登记视频不视作已交付。', '']
    videos=load(root/'videos/manifest.json')
    for video in videos.get('videos',[]):
        lines.append(f"- [{video.get('name',video.get('path'))}]({video.get('path')})：{video.get('description','')}；复跑={video.get('rerun','未标明')}；核对={video.get('verified','未核对')}。")
    if not videos.get('videos'):lines.append('当前尚无已登记视频清单。')
    lines += ['', f'核心数据：[80回合CSV](formal_readout/episodes.csv)、[总汇JSON](formal_readout/SUMMARY.json)、[真实重抓转换表](formal_readout/transitions.csv)、[冻结清单]({freeze_dir}/freeze.json)、[预登记40物体与80回合计划]({freeze_dir}/plan.json)、[资格审计](audit/settled_qualification.json)、[正式输入与判据核验](audit/final80_validation.json)、[历史引用审计](history_exposure.json)、[视频索引](videos/index.html)、[复核与复跑说明](REPRODUCE.md)。逐例JSON在`formal_readout/`；实际启动命令、环境与提交在各回合`launch.json`中。旧俯视差异和跨网格重复见证仅用于说明对应批次整批作废的原因，见[audit/initial_overhead_pair_check.json](audit/initial_overhead_pair_check.json)与[audit/geometry_tessellation_crosscheck.json](audit/geometry_tessellation_crosscheck.json)。']
    concurrency = load(root / 'formal_runs/contract.json').get('concurrency', 1)
    lines += ['', '每例`launch.json`保存精确命令、配置和环境；`batch_result.json`、`compact.json`逐例落盘用于断点续跑。以下为冻结版本复跑入口，实际路径以冻结文件和launch记录为准：', '', '```bash',
              f"cd {shlex.quote(str(LAB))}",
              f"/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python {shlex.quote(str(HERE/'stage13_batch.py'))} --root {shlex.quote(str(root/'formal_runs'))} --plan {shlex.quote(str(freeze_dir/'plan.json'))} --controller {shlex.quote(str(freeze_dir/'submitted.py'))} --concurrency {concurrency} --recording-mode minimal --resume",
              f"/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python experiments/rax_dexbin_tinyobj/stage13_readout.py --root {shlex.quote(str(root/'formal_runs'))} --output {shlex.quote(str(readout))}",
              f"/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python experiments/rax_dexbin_tinyobj/stage13_report.py --root {shlex.quote(str(root))} --readout {shlex.quote(str(readout))} --freeze {shlex.quote(str(freeze_dir))} --output {shlex.quote(str(output))}",
              '```', '',
              '普通运行默认串行，最多两路；实际并发设置以当前正式批次contract、启动审计和每例launch为准。旧`117244a`几何预跑在两路启动时出现一例原生仿真异常，随后回退串行；该异常记录包含文件监视器errno28和原生std::system_error，根因尚未确定，不能据此声称磁盘已满。该旧异常不充当当前冻结版本的基础设施失败。并发与共享主机调度影响墙钟，时间值描述本次仿真环境。原始帧和冗余日志只在报告、哈希、逐例记录及引用视频核验后清理，关键计算链所需最小轨迹保留；清理清单另存，禁止先清理再声称证据完整。', '',
              '## 论文基线判断', '']
    if not (complete and qa_complete):
        lines += ['所选版本的正式80回合与原始证据QA尚未同时完成，目前不作完整40对象成功率、抗噪性或跨手zero-shot基线结论。历史a1bbe71轨迹证明该旧版本连续三抓曾实际下发，方块/圆柱固定三指当时仍为规划拒绝；这不验收新增守卫的V4。最终按全部对象保留失败、零三指执行与零救回。']
    else:
        clean=groups['clean'];noise=groups['noise']
        three=clean.get('attempt_fingers',{}).get('3',0)+noise.get('attempt_fingers',{}).get('3',0)
        lines += [f"原视觉首抓{clean['first_successes']}/40，三抓最终{clean['successes']}/40；加噪首抓{noise['first_successes']}/40，三抓最终{noise['successes']}/40。三抓相对首抓新增分别为{clean['successes']-clean['first_successes']}和{noise['successes']-noise['first_successes']}个对象；噪声后的最终成功数变化为{noise['successes']-clean['successes']:+d}/40。",
                  f"正式实际三指尝试{three}次，重抓救回{summary.get('rescues',0)}回合。" + ('本结果应称为两指抓取基线；auto评估过三指候选，但证据未支持三指执行能力或换指组救回。' if three==0 else '三指执行、最终成功与换指组救回须分别由逐例证据判断，不能由auto配置名推断。'),
                  '恢复后成功是连续回合中的救回记录；没有无恢复消融，不能据此证明某个恢复动作的因果增益。每物体仅一个初态和噪声种子，普通实时FingerEye渲染仍有微小变化，不能把每个不一致结局全部归因于高斯噪声；本轮也没有验证跨手zero-shot。该冻结协议可以作为下一轮不同机械手对照的候选协议，能力范围、失败分布和噪声差异须随完整表格一起报告。']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')
    (output.parent/'history_exposure.json').write_text(json.dumps(history,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output.parent/'timing_audit.json').write_text(json.dumps(timing_audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(output),'complete':complete,'episodes':summary.get('episodes',0),'object_rows':len(all_objects)},ensure_ascii=False))


if __name__=='__main__':
    main()
