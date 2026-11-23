Stage16 数值与两指/三指配对证据

本包从既有物理回合独立重读、重算，没有新增物理运行。
这是两个预先登记的开发场景，每条件一次；不能当成随机物体总体成功率。
原 two/three/auto 六回合正式4/6；另含 normal4 与 bounded_relative 两个独立
单因素对照后，正常入口10回合正式6/10。四个three回合均0步、0闭合可行性拒绝，
仍保留在分母内。最佳安全三指诊断20_safe单列，不并入正常入口。

共同初态与边界
物体20/34，每场景共用初始普通RGB、实测初始q、种子与物理条件；后续RGB各自
真实闭环读取。具体输入hash、实测初态、物理参数见scenario_registration.json和
paired_scenarios.json。在线未使用物体真值位姿、mesh或接触力；独立复核读取它们。
没有真机运行，hardware n=0。

数值结论
1. 最终固定目标IK的位置和法向残差都乘sqrt(2/n)。法向是单位向量弦差，
   不直接以角度作为残差；角度theta以rad保存。
   SciPy法向cost = w^2 mean[2(1-cos(theta_i))]。
   二指w4、三指w7的等误差贡献之比仍是49/16=3.0625；共同w4可消去此差。
   此权重作用于生成阶段，不是最后的任务裕量排序。
2. 两个真实固定输入案例的原三指w7最终q与历史保存值逐数相同；只改最终权重
   为4会改善位置误差、部分法向变差，仍未通过正常门槛。区域IK和硬门槛未改。
   candidate_table.csv给出12行，fixed_candidate_audit.json保留目标、q、FK、
   每指误差、各平方项、权重、归一化、关节余量、门槛及共同力学假设。
3. 旧排序平局带包含nominal-robust差。该差是模型情景敏感性，不是概率置信带。
   合成算术反例：two名义1.8/稳健.8/2s，three名义1.5/稳健1.2/4s。
   旧带1.0选two；修订相对带.12选three。该反例没有物理成功声明。
   本批实际合法池只有two，固定合法候选新旧排序完全一致；没有三指排序收益。
4. 同一质量5–15g、摩擦下界.3、扰动.02N/.0001Nm、每垫1N、执行器限值和
   RGB中心假设下计算LP。LP不是测得真机承载证书；Jacobian、负载与最坏扰动
   见jacobian_and_worst_loads.json和完整候选JSON。

真实闭环与承载
正常two与auto在20/34上的完整steps.jsonl hash分别一致，均真实闭合1次并保持成功。
每次正常three搜索36个布局/初值，未触发42s预算；没有双组同时可行的正常池。
20 w7拒绝：法向27、位置7、端点2；w4：30、4、2。
34 w7拒绝：法向27、位置8、端点1；w4：32、3、1。
每指max和整段路径硬门槛始终保留，RMS没有替代单指可达或碰撞判断。

20_safe为隔离RGB包络误拒诊断，另经实际几何守卫，正式/legacy_strict均成功。
末2s 20Hz40样本：thumb/index指定垫各40/40承载；middle指定垫0/40；
middle component1壳40/40承载约.00247–.00259N。故不是正常三个指垫成功。
原120Hz thumb/index原生实际接触点可读，末窗各240/240承载；middle和壳实际点
位置未保存。physical/native_contact_points_summary.json明确采样率和缺字段范围。
不把FK垫中心、最近mesh点或连杆中心写成实际接触点。

计时与轨迹结论的范围
paired_modes.csv同时保留物理时间、result elapsed和完整启动墙钟；启动墙钟可能
含资源准入等待。三者不是同一边界，也不是常驻真机任务至稳定抓持的时延。
20 auto_tie比two少900步，但共同3294步指令/实测q/物体状态相同、候选顺序相同，
后续RGB支撑判断不同；首次像素差为step8/frame2各1通道1DN，不能据此认定因果。
因此不把少900步计为排序提速。34 two/auto/tie完整轨迹hash相同。
本包不声称40秒真机目标达成。

文件与独立复算
paired_modes.csv：11回合，含10个正常入口及独立诊断、0闭合拒绝和所有原分母。
per_finger_errors.csv：实际最佳合法two与明确标注的被拒three的每指数值。
ik_residual_contributions.csv：各残差平方和及SciPy cost贡献，统一位置m/角rad。
physical/actual_best_candidate_details.json：真实入口的中间值与硬门槛。
physical/fixed_feasible_ranking_ab.json：固定合法池旧新排序及平局带。
physical/physical_readout.json：评分、抓次、载荷、版本、公共输入与raw hash。
physical/image_divergence.json和final_decision_divergence.json：首差与后继判断。
normal_and_tie_witnesses.json：等误差权重公式及明确标合成的排序算术反例。
AUDIT.json：新一次独立120Hz原始记录重裁决与hash核对、源/派生文件hash。
这次重裁决不算新增实跑。源码版本也不会被误写为历史每回合执行版本。

在包含本地原始日志的完整工作树，从experiments/rax_dexbin_tinyobj运行：
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYTHONDONTWRITEBYTECODE=1 python publication/stage16_20261122/numerics/build_public.py
运行需要该项目IsaacLab Python环境中的numpy/scipy；脚本不启动仿真。
可用--source-here指定上述实验目录，--output指定全新导出目录。

仿真复跑示例（新输出目录，普通入口）：
python run_stage16.py --session ../../outputs/stage16_NUMERIC_NEW/20_two \
 --case-file stage16_evidence/numerics/cases/20_two.json \
 --evaluation-mode deferred --mode minimal --fingers two
将two改three或auto、20改34。独立权重对照只在three追加--normal-weight 4；
排序对照只在auto追加--margin-tie bounded_relative。保留默认已验证two配置。
全部原执行argv在physical/commands.json，<LOCAL_DOCS>/<LOCAL_HOME>为本地路径占位；
本包不发布相机像素、录像或逐物理步大日志，缺少本地原数据时不能重构这些数据。
