from pathlib import Path
import csv,json,re
R=Path(__file__).resolve().parents[2];E=R/'stage17_evidence';rows=list(csv.DictReader((E/'lighting_episodes.csv').open()));valid=[r for r in rows if r['selected']=='True' and r['execution_valid']=='True'];assert len(valid)==18;assert len(rows)==21;assert sum(r['execution_valid']=='False' for r in rows)==3
base=[r for r in valid if r['name'].endswith('_legacy')];hs=[r for r in valid if r['cohort']=='lighting_v2'];v1=[r for r in valid if r['cohort']=='lighting' and r['name'].endswith('_hs')];assert len(base)==len(hs)==len(v1)==6
pairs=list(csv.DictReader((E/'lighting_pairs.csv').open()));assert len(pairs)==6
lines=['|真实灯强|Stage16原版正式|H/S v2正式|S→F|F→S|双方成功|双方失败|','|---|---:|---:|---:|---:|---:|---:|']
for light in (.4,1.,1.6):
 ps=[r for r in pairs if float(r['light'])==light];lines.append(f'|{light:g}|{sum(r["baseline"]=="True" for r in ps)}/2|{sum(r["hs_v2"]=="True" for r in ps)}/2|{sum(r["transition"]=="SF" for r in ps)}|{sum(r["transition"]=="FS" for r in ps)}|{sum(r["transition"]=="SS" for r in ps)}|{sum(r["transition"]=="FF" for r in ps)}|')
lines+=['','正式总数：原版5/6，H/S v2 6/6；成功→失败0，失败→成功1（B、灯强0.4）。两版首抓均3/6，B均首抓失败后执行第二次闭合；实际闭合各9次，单回合最多2次，未突破最多3次预算。双方全程几何均6/6；旧strict均3/6，旧结果并列保留，没有评分版本变更。','','|用例/灯强|原版→H/S|实际闭合原/新|物理秒原/新|host秒原/新|launch秒原/新|末窗最小lift mm原/新|','|---|---|---:|---:|---:|---:|---:|']
for light in (.4,1.,1.6):
 for case in ('A','B'):
  a=next(r for r in base if r['case']==case and float(r['light'])==light);b=next(r for r in hs if r['case']==case and float(r['light'])==light);lines.append(f'|{case}/{light:g}|'+('成功' if a['success']=='True' else '失败')+'→'+('成功' if b['success']=='True' else '失败')+f'|{a["grasps"]}/{b["grasps"]}|{float(a["physical_s"]):.2f}/{float(b["physical_s"]):.2f}|{float(a["host_s"]):.2f}/{float(b["host_s"]):.2f}|{float(a["launch_s"]):.2f}/{float(b["launch_s"]):.2f}|{float(a["lift_mm"]):.3f}/{float(b["lift_mm"]):.3f}|')
lines+=['','B三档两版第一处实际失败均在首抓试抬/最终视觉决策step3450（28.75物理秒）：在线`retry/observed_failure`，末窗实际lift约−0.100mm；随后同回合下放/松指并执行重抓。暗光原版最终`stop_unknown`，末窗27.730020mm，低于28mm正式下限0.269980mm；H/S v2重抓后29.389161mm通过。A三档H/S最终均`stop_unknown`，但离线正式均通过：在线保留未知并停止继续破坏可能已携带的抓持，不等于正式评分失败。逐次在线动作/原因和正式指标在lighting_diagnostics.csv。','','收益限于本批观察：数字V×0.4–1.6以及同一原始初始俯视RGB的legacy/HS取色都没有分割差异（见initial_mask_audit.json）；因此不能把B暗光的F→S直接归因于移除V>35。六路后续可靠域、配准、恢复路径及微小渲染差异的贡献未被单因素消融分离。基线B正常光冻结复跑50.55物理秒，H/S为70.50秒，增加19.95秒；原版/新版本六回合总物理时间291.75/301.55秒，**本批成功多1例但没有整体提速**。','','共实际执行21回合：原版/H/S初版各6个注册条件，H/S v2 6个，另3次完整性复跑。开发时修改/新增顶层源码使B正常光原版、B正常光H/S初版、A暗光原版3回合完整性失效；物理结果虽成功，全部排除正式分母，保留原记录，源码冻结后新建immutable_rerun，3次均完整性和正式通过。有效选用18回合；版本比较仅用原版与v2的12回合。H/S初版闭环也为6/6（各灯强2/2），但其数字强光反例已明确证伪该裁剪规则，保留作开发证据，不推荐。未把旧视频/离线重复读算作新物理试验。','','[逐回合表](lighting_episodes.csv)、[六对转换表](lighting_pairs.csv)、[原始记录SHA与独立评分](lighting_readout.json)、[逐次决策](lighting_diagnostics.csv)提供所有结果，包括失败和排除记录。']
block='\n'.join(lines)
recommend='''本批推荐 **H/S v2 + 清理后的共用实现，fast/two/max3/preload-off**，保留Stage16冻结原版作为可复跑基线。理由是同批6/6通过、无S→F，并修复极暗回放的V下限问题和初版强光裁剪错误；范围仅是本批两个开发物体和三个灯强。正常光B耗时增加，未证明总体提速或总体泛化率提高。

|项目|最终状态|证据与边界|
|---|---|---|
|实际版本/入口/哈希/最小核心表|实际通过|Stage16完整树核对；正常入口生成v2，SHA一致；用户未提交工作未改|
|核心冗余清理|实际通过（已测路径）|A/B命令精确相等、120几何裁决相等、11项检查通过；无提速声明|
|同RGB配对回放|实际通过；初版实际失败已修正|0.4–1.6 v2无新增退化；0.15定位旧V下限；非物理成功率|
|真实同色改光闭环|实际通过（本批范围）|原版5/6、v2 6/6；暗/正常/亮各2回合，包含首抓和重抓；本批保成功，未证明总体泛化|
|门槛误拒核查|复核实际通过；可合入的几何/评分放宽无充分证据|20保守包络有见证；34真实冲突不能全放；D具体阻塞是cooked/authored一致性；高度不足保留失败|
|中指接触图与分类|实际通过；正常三指定垫实际失败|掌侧component1正载荷40/40，指定component3 0/40；贡献份额未消融|
|真正外部侧视/全程肉眼无滑落|具体阻塞|原相机被掌遮挡，没有真正侧视；可见帧与毫米几何分开报告|
|公开交付|发布后由远端读回逐文件核对|明确选集约11.4MB；资产/原始大日志留本地，固定提交见发布回执|

下一轮优先项是给B增加重复种子/初态以区分视觉收益与回合变动、补真正外部侧视、统一D碰撞几何；正常三指安全包络只在有额外可靠视角且不借真值的条件下研究。本轮不再用孤例扩大门槛改动。'''
p=E/'REPORT.md';s=p.read_text()
for name,body in [('LIGHTING_RESULTS',block),('RECOMMENDATION',recommend)]:
 pattern=r'<!-- '+name+r' -->(?:.*?<!-- END_'+name+r' -->)?';s=re.sub(pattern,'<!-- '+name+' -->\n\n'+body+'\n\n<!-- END_'+name+' -->',s,flags=re.S)
p.write_text(s)
# Parallel online and official verdicts without claiming the private score is observable.
details=json.loads((E/'lighting_readout.json').read_text());diags=[]
for d in details:
 for i,a in enumerate(d['attempts']):
  decision=a['decision'];diags.append(dict(name=d['name'],selected=d['selected'],execution_valid=d['execution_valid'],episode=d['episode'],decision_index=i+1,actual_grasps=a['actual_grasps'],step=a['decision_step'],time_s=a['decision_step']/120,online_action=decision.get('action'),online_reason=decision.get('reason'),support_state=decision.get('support_state'),prefix_official_success=a['prefix_official_success'],prefix_geometry_passed=a['prefix_geometry_passed'],terminal_lift_mm=a['lift_m']*1000 if a['lift_m'] is not None else None,terminal_clearance_mm=a['clearance_m']*1000 if a['clearance_m'] is not None else None,whole_episode_official_success=d['success']))
with (E/'lighting_diagnostics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(diags[0]));w.writeheader();w.writerows(diags)
print('final report: primary 12 valid, baseline 5/6, hs v2 6/6, development v1 6/6, excluded 3')
