"""Fetch both remote branches afresh and check immutable source/evidence bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parents[2]
ROOT=HERE.parents[2]

def run(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True).strip()

def sha(blob):return hashlib.sha256(blob).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-commit',required=True)
    p.add_argument('--private-branch',default='fingereye-clean-mainline')
    p.add_argument('--public-receipt',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'mainline/publish.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--work-root',type=Path)
    a=p.parse_args();work=a.work_root or Path(tempfile.mkdtemp(prefix='fingereye-clean-readback-'))
    work.mkdir(parents=True,exist_ok=True)
    config=json.loads(a.config.read_text());receipt=json.loads(a.public_receipt.read_text())
    pub=receipt['repositories']['research'];assert pub['pushed']
    private=work/'private.git';public=work/'public'
    run(['git','clone','--quiet','--bare','--depth','1','--filter=blob:none','--single-branch','--branch',a.private_branch,'https://github.com/ztr583/grasp_small_obj.git',str(private)])
    pc=run(['git','rev-parse','HEAD'],private);assert pc==a.private_commit,(pc,a.private_commit)
    run(['git','clone','--quiet','--depth','2','--single-branch','--branch','main',pub['remote'],str(public)])
    uc=run(['git','rev-parse','HEAD'],public);assert uc==pub['commit']
    manifest=json.loads((public/config['research']['manifest_path']).read_text())
    assert manifest['source_revision']==receipt['source_revision']
    # Compare entire private tree by Git object IDs: private assets need no checkout.
    private_tree=run(['git','rev-parse',pc+'^{tree}'],private)
    local_tree=run(['git','rev-parse',a.private_commit+'^{tree}'],ROOT)
    assert private_tree==local_tree
    names=set(run(['git','ls-tree','-r','--full-tree','--name-only',pc],private).splitlines())
    prefix='FingerEyeLab/experiments/rax_dexbin_tinyobj/'
    deleted=json.loads((HERE/'mainline/evidence/deletions.json').read_text())
    for row in deleted['removed_files']:assert prefix+row['path'] not in names,row['path']
    selected=[]
    for dest,source in config['research']['files'].items():
        actual=(public/dest).read_bytes();local=(ROOT/source).read_bytes()
        assert sha(actual)==sha(local)==manifest['files'][dest]['sha256'],dest
        selected.append(dict(path=dest,sha256=sha(actual)))
        if dest.endswith('.py'):compile(actual,str(public/dest),'exec')
    old=[n for n in run(['git','ls-files'],public).splitlines() if n.startswith('studies/stage17_20261123/')]
    assert old==['studies/stage17_20261123/ARCHIVE.md'],old
    for name in ['mainline/strategy.py','run_stage17.py','general_grasp_contacts.py','evaluator.py','round7c_dynamics.py']:
        raw=subprocess.check_output(['git','show',pc+':'+prefix+name],cwd=private)
        assert sha(raw)==sha((HERE/name).read_bytes()),name
    study=public/'studies/fingereye_clean_mainline_20261123'
    code="from pathlib import Path; import hashlib; from mainline import strategy; from run_stage17 import controller_settings; p=Path(strategy.__file__); assert controller_settings(p)['S17_COLOR_MODE']=='hs'; print(hashlib.sha256(p.read_bytes()).hexdigest())"
    source_sha=run([sys.executable,'-c',code],study)
    expected=sha((HERE/'mainline/strategy.py').read_bytes());assert source_sha==expected
    # Preserve all prior public files outside this export and explicit pruning.
    # Historical report directories may already have been retired before this release.
    previous=run(['git','rev-parse',uc+'^'],public)
    def tree(revision):
        raw=subprocess.check_output(['git','ls-tree','-r','-z',revision],cwd=public)
        return {entry.split(b'\t',1)[1].decode():entry.split(b'\t',1)[0].split()[2].decode()
                for entry in raw.split(b'\0') if entry}
    before=tree(previous);after=tree(uc)
    exported=set(config['research']['files'])|{config['research']['manifest_path']}
    prune=config['research'].get('prune_prefixes',[])
    retained=[n for n in before if n not in exported
              and not any(n==p or n.startswith(p+'/') for p in prune)]
    assert all(after.get(n)==before[n] for n in retained), 'Unlisted prior public file changed'

    report=dict(private_remote='https://github.com/ztr583/grasp_small_obj',private_branch=a.private_branch,private_commit=pc,
                private_tree_sha=private_tree,private_tree_matches_local=True,removed_private_paths_absent=deleted['removed_file_count'],
                public_remote=pub['remote'],public_commit=uc,public_source_revision=manifest['source_revision'],
                selected_files_verified=len(selected),all_selected_bytes_match=True,public_strategy_import_sha256=source_sha,
                public_old_stage17_only_archive=True,previous_public_commit=previous,
                preserved_unlisted_files=len(retained),all_unlisted_prior_files_unchanged=True,
                private_checkout=str(private),public_checkout=str(public),verified_files=selected)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='verified_files'},indent=2))

if __name__=='__main__':main()
