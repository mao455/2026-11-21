Stage16 private all-link native contact capture: new physical diagnostic

One new isolated physical run completed, returncode0, no failed launch.
Base60b5a7b007f. Only evaluator-private stage2_contact_audit.py changed in isolated
fingereye_stage16_contact_closeout worktree. Public controller and evaluator unchanged.
Controller SHA c38a5fd98d526e97cc8189b4c225f63ab93a8c611c1ba2ff5d0a33f5d72180eb.
Same20_three case and verified initialRGB as prior best-safe diagnostic. Original
S14_DIAGNOSTIC_STOP=1 guard active, inline geometry; mode minimal. This isolated
RGB-envelope intervention is diagnostic, not normal three-finger policy acceptance.

Result: formal success=true, legacy_strict=true, geometry=true, one actual closure.
3816 actual physics states,31.8 physicals, result elapsed83.039848s. No hardware run.
Entire steps.jsonl hash identical to original Stage16 20_safe diagnostic: true.
Independent raw state count/timestamps/scorer/controller verification passed.
Minimum hold lift30.04635mm, clearance29.76469mm, commanded lift30.00000mm,
continuous terminal hold2s. First geometry failure: none.

Native all-link sidecar cadence20Hz,40 observations over final2s,step3576..3810.
Middle component3 pad:0/40 loaded samples,0 native contacts.
Middle component1 camera housing:40/40 loaded samples, .00246612–.00258846N.
48 native proximity manifold points per sample, but exactly one >1mN loaded point
per sample. Zero-force manifold points are retained and clearly separate from loads.
Loaded actual world point bounds(m): [[0.01793375052511692, 0.003416412975639105, 0.07471740245819092], [0.017933839932084084, 0.0034165806137025356, 0.07471750676631927]]
Loaded native separation 0.000104308..0.000158325mm.
Thumb/index component3 pads each40/40, .309887–.310090N/.306591–.306884N.
Classification: housing-assisted three diagnostic; normal three pad accepted=false.
No missing actual middle housing contact position remains for this new20Hz run.
No missing middle pad point is invented: it had zero physical contact in this window.

hold_native_contacts.csv preserves all2280 native points, includingzero force,
with step,time_s,phase,link,point index, worldXYZ(m),unitnormal,force(N),separation(mm).
actual_contacts_summary.json distinguishes all proximity points and >1mN loaded
points, stores exact sampledsteps,contactcounts,source/rawhashes and formal score.
Perpoint and joint snapshots are copied before any future native buffer query.
Native engine truth remains only in private evaluator sidecar; original
super().read(dt) result returned unchanged, no new controller input.

Patch: evaluator_private.patch; exact patchedsource retained. Apply to an isolated
worktree at60b5a7b007f with git apply --unidiff-zero, then run COMMAND.txt. Keep case reference
and controller bytes as recorded. For relocating, set original case's initial RGB
reference to copied initial_rgb20.npy/20.json and preserve their hashes.
read_capture.py --session SESSION --experiment-here EXPERIMENT_DIR reruns offline
verification; requires the complete local raw session, not only the public export.
RawRGB,perphysicalstep logs,fullsidecar and backend log remain local in
<LOCAL_DOCS>/fingereye_stage16_contact_closeout/FingerEyeLab/outputs/stage16_contact_closeout/20_safe_points.
This evidence adds one diagnostic only; it does not change the normal10 denominator
or frozen defaulttwo. It closes the actual contactpoint capture gap at20Hz, not120Hz.
