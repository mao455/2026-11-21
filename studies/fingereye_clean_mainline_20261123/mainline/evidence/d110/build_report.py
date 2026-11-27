"""Prepare delivery only after ten original fixtures finish on one source."""
from pathlib import Path
import collections,json,shlex
E=Path(__file__).resolve().parents[3];R=E.parents[2];H=E/'mainline/evidence/d110';O=R/'FingerEyeLab/outputs/d110';W=O/'maintenance/finite_retry_worktree'
SOURCE='058c570113ba868b0213fed19f19d2e5c434789d'
def main():
 audit=json.loads((H/'V53_FINAL_RESULTS.json').read_text());state=json.loads((H/'V53_FINAL_STATE.json').read_text());rows=audit['rows']
 assert len(rows)==10 and state['source_commit']==SOURCE
 required={('xhand','box_center'),('xhand','cylinder_offset'),('xhand','wedge_offset'),('new','box_center'),('new','cylinder_offset'),('wuji','cylinder_28')}
 assert all(r['within_three_success'] and r['first_grasp_success'] for r in rows if (r['hand'],r['case'])in required),'An old first success regressed; do not release'
 assert required <= {(r['hand'],r['case']) for r in rows}
 assert all(r['source_unchanged'] and r['actual_closures']<=3 and r['geometry_valid_every_step'] and r['continuous_120hz'] for r in rows)
 stats={}
 for hand in ('xhand','new','wuji'):
  part=[r for r in rows if r['hand']==hand]
  stats[hand]=dict(cases=len(part),first=sum(r['first_grasp_success']for r in part),within_three=sum(r['within_three_success']for r in part),zero=sum(r['actual_closures']==0 for r in part),rescued_second=sum(r['success_on_closure']==2 for r in part),rescued_third=sum(r['success_on_closure']==3 for r in part))
 summary=dict(source_commit=SOURCE,normal_entry_only=True,stats=stats,rows=[])
 lines=['D110 三手共享抓取物理复验',f'执行代码：{SOURCE}','原十例；120/20/30 Hz；最多三次实际闭合；抓间不重置；原 evaluator。','诊断回放不计正式成绩。','']
 for hand,v in stats.items():lines.append(f"{hand}: 首抓 {v['first']}/{v['cases']}；三抓内 {v['within_three']}/{v['cases']}；零闭合 {v['zero']}；第二/第三抓救回 {v['rescued_second']}/{v['rescued_third']}")
 lines+=['','hand | case | 首抓 | 闭合次数 | 三抓内 | 最低保持抬高mm | 规划CPU秒 | 回合墙钟秒']
 for r in sorted(rows,key=lambda r:(('xhand','new','wuji').index(r['hand']),r['case'])):
  p=r['physics'];item={k:r[k]for k in ('hand','case','session','actual_closures','first_grasp_success','within_three_success','success_on_closure','controller_completed','controller_error','endpoint_refusals','recovery','stops','planning_cpu_max_s','trace_elapsed_s','requested_simulation_s_by_phase')}
  item['physics']={k:p.get(k)for k in ('success','geometry_passed','first_invalid_step','hold_min_lift_m','hold_duration_s','failure')};summary['rows'].append(item)
  lines.append(f"{r['hand']} | {r['case']} | {r['first_grasp_success']} | {r['actual_closures']} | {r['within_three_success']} | {p.get('hold_min_lift_m',0)*1000:.3f} | {r['planning_cpu_max_s']:.2f} | {r['trace_elapsed_s']:.2f}")
  if not r['within_three_success']:lines.append('  失败实况：'+json.dumps(dict(refusals=r['endpoint_refusals'],stops=r['stops'],last_recovery=r['recovery'][-3:],error=r['controller_error']),ensure_ascii=False))
 lines+=['','主要修改与证据：','1. 原始碰撞网格、运动链和固定装配生成外露接触区域；等价末节坐标变换在13个姿态下48个区域覆盖一致，最大世界点误差3.99e-17m。窄面、曲面保留小接触斑。','2. 初始快速搜索保留静态/已验证指面，有限RGB面内重定位与联合区域搜索补充；承重与抗扭收益决定第三指是否值得加入。','3. 接近可行的IK从自身解重求解，位置门槛仍0.8mm；冻结圆柱输入从1.255/1.241mm拒绝降至约4.11e-8m。第一段闭合与后续闭合同样允许未参与夹持的关节避让，完整检查实测连接、限位、桌面和真实自碰。','4. FIRST_CLOSE_IDLE_ROUTE_PAIR.json：同一抓姿q最大差0，同一公开几何和合法起点，旧初始闭合在第4/5指中节自碰，新整手路径通过。这是离线路径证据。','5. SAME_CANDIDATE_PHYSICAL_PAIR.json：相同归档公开候选、同一公开几何、各自重新生成整手路径，两个执行器均合法成功，最低保持抬高30.858/31.208mm。说明正常圆柱退步先发生在IK候选准入；诊断脚本首次漏color的报错回合不参与此结论或成绩。','6. 逐指小步闭合保留持续编码器偏差、停稳检测和有限累计探闭；每步按实际支持半径限制，实测指关节速度按URDF限值核验。没有统一放宽碰撞余量。','7. 视觉目标变化、IK残差、指令与实测q偏差分别记录；接触面/法向变化也触发局部修正；无变化时保留已检查路径。','8. 首抓与救回分开统计；RGB无法确认空抓时会保留可能已持物的姿态，不能用离线真值授权线上重抓。','', '复跑命令：见 evidence/d110/REPRODUCE.txt；新增物体示例见 TEACHER_CASE.md。']
 lines+=['','证据边界与剩余失败：','- 正式十例全部由同一执行提交 058c570113ba868b0213fed19f19d2e5c434789d 的正常入口运行；源码、三份静态描述和共享相机参数冻结。八个成功全部为一次闭合，均连续合法保持2秒，最低实际抬高28.828–32.167mm；十例全程几何有效。原六个首抓成功均未退步。\n- 新增近解重试和初始闭合闲置指避让在精确离线路径对照中有效；它们没有在八个正常成功的最终选中候选上触发，不能把八例成功归因于这两个最后加入的辅助修正。正式增益来自此前已合入的外表面覆盖、联合区域搜索与有界闭合等共享改动；没有完成这些改动逐个独立消融。\n- 五边柱实际选中的两个源接触来自固定装配的 FingerEye 壳体外表面。D109中法向相近的最近旧候选分别距它们16.799和20.754mm；新接触支持半径分别2.000715和0.879199mm。见 PENTAGON_SOURCE_COVERAGE.json。原生接触日志按物理导入后固定装配的末节 link 记录，保持段两指均接触240/240步；不能仅凭该link名断言是哪块三角面承重。\n- Wuji楔体：第一次闭合有真实接触，但最大实际抬升仅0.05516mm，物体水平移动约0.792mm。选中方案在保守摩擦/支撑模型下的任务余量为0，策略仍将其作为受保护试探执行；此次未形成足以承重的抓持。仅凭现有记录不能把原因唯一归为摩擦、落点或闭合行程。策略在线确认失败，实际退让、重观察并取得有效恢复票据，未重置物体。第二轮确实用了106.660 CPU秒搜索10条直接候选，0条通过；不是没有识别重抓机会。首条在避碰细化后两指位置误差5.887/3.862mm、桌面净空-0.346mm。后续联合区域扩展受累计CPU时限约束，本轮未做改变该调度的配对验证，不能宣称剩余状态无可行抓姿。实际闭合总数1，第二/第三抓均无救回。\n- 原三指手楔体：首次搜索35条候选均未完成IK和整手路径。首条位置误差4.359/3.667mm、法向误差0.556/0.669rad、桌面净空-0.239mm；总拒绝为位置28、法向4、端点真实碰撞2、接近/闭合路径1。到规划CPU余量停止时仍为0次实际闭合，属于规划未解，不能作为闭合物理失败或证明不可抓。\n- 主要耗时：规划CPU心跳峰值87.663–528.255秒；回合控制轨迹墙钟349.821–1065.975秒。Wuji盒体345.916/723.596秒、五边柱301.362/846.822秒、楔体437.117/1065.975秒；原三指楔体528.255/742.977秒。心跳峰值不是精确的最终总CPU；各次规划的实际耗时见原始trace。\n- 本轮正式成功均为两指；三指参与可达性/力臂预筛选、独立IK与同姿态两指子集承重比较，但本批没有三指胜出并成功的物理证据。摩擦和力限为显式仿真假设；编码器阻滞只是代理，不作真实接触力观测。\n- 59项相关单元检查通过；13姿态等价坐标系回放通过。未完成两例楔体，不以诊断成功或旧版本结果替换。原始日志与RGB继续保存在 FingerEyeLab/outputs/d110/ 下；本证据选集记录路径和哈希。']
 summary['limitations']=['Two wedge cases remain unsolved', 'All eight normal successes use two contacts; no demonstrated triad win', 'Near-contact retry and initial idle-route repair are offline causal evidence, not triggered helpers in eight successful selected normal candidates', 'Wuji wedge retry actually searched ten direct candidates, but expanded search had limited cumulative CPU; reachability is not disproved']
 summary['baseline_first_successes_preserved']=6
 (H/'FINAL_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');(E/'mainline/D110_REPORT.txt').write_text('\n'.join(lines)+'\n')
 commands=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(R))}','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1']
 for row in state['rows']:
  args=[x.replace(str(W),str(R))for x in row['argv']];args[args.index('--session')+1]=str(R/'FingerEyeLab/outputs/d110/reproduce'/f"{row['case']}_{row['hand']}")
  commands+=['',shlex.join(args)]
 (H/'REPRODUCE.txt').write_text('\n'.join(commands)+'\n');print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
