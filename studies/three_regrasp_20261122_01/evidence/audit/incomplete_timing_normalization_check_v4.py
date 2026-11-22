"""Exercise offline normalization against the five real frozen-V4 receipts."""
import contextlib
import copy
import hashlib
import io
import json
import pathlib
import shutil
import sys
import tempfile

LAB = pathlib.Path(__file__).resolve().parents[3]
ROOT = LAB/'outputs/three_regrasp_20261122_01'
sys.path[:0] = [str(LAB/'experiments/rax_dexbin_tinyobj'), str(ROOT/'reproduction')]
from stage13_readout import aggregate
from normalize_incomplete_timing import normalize, RAW_FIELD

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

names = [r['name'] for r in load(ROOT/'audit/final80_partial_first5_v4.json')['episodes']]
assert len(names) == 5
temporary = pathlib.Path(tempfile.mkdtemp(prefix='fingereye_timing_normalization_first5_v4_'))
batch = temporary/'batch'
batch.mkdir()
for name in names:
    (batch/name).symlink_to(ROOT/'formal_runs'/name, target_is_directory=True)
(batch/'contract.json').symlink_to(ROOT/'formal_runs/contract.json')
(temporary/'frozen_v4').symlink_to(ROOT/'frozen_v4', target_is_directory=True)
protected = [ROOT/'formal_runs'/name/filename for name in names for filename in ('compact.json','batch_result.json','episode/steps.jsonl','episode/agent_trace.jsonl','episode/image_manifest.jsonl')]
protected_before = {str(p): sha(p) for p in protected}
raw = temporary/'raw_readout'
with contextlib.redirect_stdout(io.StringIO()):
    aggregate(batch, raw)
raw_files = {p.name: sha(p) for p in raw.iterdir() if p.is_file()}
output = temporary/'normalized_readout'
manifest = normalize(raw, output, temporary/'copy_mode_audit.json')
again = temporary/'idempotent_readout'
again_manifest = normalize(output, again, temporary/'idempotence_audit.json')
inplace = temporary/'inplace_readout'
shutil.copytree(raw, inplace)
snapshot = temporary/'inplace_snapshot'
inplace_manifest = normalize(inplace, inplace, temporary/'inplace_audit.json', True, snapshot)
before, after = load(raw/'02_abc_76395_clean.json'), load(output/'02_abc_76395_clean.json')
source_summary, normalized_summary = load(raw/'SUMMARY.json'), load(output/'SUMMARY.json')
checks = dict(
    five_real_receipts_used=set(names)=={r['name'] for r in source_summary['results']},
    copy_mode_internal_checks_passed=manifest['all_checks_passed'],
    exactly_one_timeout_normalized=manifest['changed_episode_names']==['02_abc_76395_clean'],
    timeout_complete_policy_time_is_null=after['policy_compute_s'] is None,
    original_zero_preserved_and_labelled=after[RAW_FIELD]==before['policy_compute_s']==0 and 'nor process CPU time' in after['policy_compute_time_scope'],
    physical_lower_bound_and_all_outcome_fields_unchanged=all(before[k]==after[k] for k in ('physical_s','first_trial_physical_s','retry_physical_s','phase_physical_s','observed_physical_lower_bound_s','physical_time_scope','success','first_success','rescued','actual_grasps','finger_counts','branch_counts','closures','transitions','integrity_valid','controller_completed','failure_class')),
    persisted_timeout_lower_bound_remains_43_5=after['physical_s']==after['observed_physical_lower_bound_s']==43.5,
    four_completed_json_files_byte_identical=all(sha(raw/(n+'.json'))==sha(output/(n+'.json')) for n in names if n!='02_abc_76395_clean'),
    summary_statistics_unchanged=all(source_summary[k]==normalized_summary[k] for k in source_summary if k!='results'),
    transition_csv_byte_identical=sha(raw/'transitions.csv')==sha(output/'transitions.csv'),
    normalization_is_idempotent=again_manifest['changed_episode_names']==[] and all(sha(output/p.name)==sha(p) for p in again.iterdir() if p.is_file()),
    inplace_matches_copy_output=all(sha(output/p.name)==sha(p) for p in inplace.iterdir() if p.is_file()),
    inplace_snapshot_preserves_original_changed_files=all(sha(snapshot/p)==digest for p,digest in inplace_manifest['source_sha256'].items()),
    source_readout_not_changed=raw_files=={p.name:sha(p) for p in raw.iterdir() if p.is_file()},
    formal_episode_receipt_and_compact_files_not_changed=protected_before=={str(p):sha(p) for p in protected},
)
result = dict(schema=1, scope='Real first-five offline normalization exercise. Temporary batch uses symlinks only; no simulations or protected writes.', names=names, temporary_directory=str(temporary), normalized_readout=str(output), manifests={k:str(temporary/v) for k,v in [('copy','copy_mode_audit.json'),('idempotence','idempotence_audit.json'),('inplace','inplace_audit.json')]}, changed_episode='02_abc_76395_clean', timeout_policy_compute_before=before['policy_compute_s'], timeout_policy_compute_after=after['policy_compute_s'], preserved_accumulator=after[RAW_FIELD], physical_lower_bound_s=after['physical_s'], group_policy_compute_stats_before={k:source_summary['conditions'][k]['timing_s']['policy_compute_s'] for k in ('clean','noise')}, group_policy_compute_stats_after={k:normalized_summary['conditions'][k]['timing_s']['policy_compute_s'] for k in ('clean','noise')}, checks=checks, all_checks_passed=all(checks.values()), script_sha256=sha(ROOT/'reproduction/normalize_incomplete_timing.py'), check_builder_sha256=sha(pathlib.Path(__file__)), protected_sha256=protected_before)
(ROOT/'audit/incomplete_timing_normalization_first5_v4.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert result['all_checks_passed'], checks
print(json.dumps({k:result[k] for k in ('temporary_directory','normalized_readout','timeout_policy_compute_before','timeout_policy_compute_after','preserved_accumulator','physical_lower_bound_s','all_checks_passed')}))
