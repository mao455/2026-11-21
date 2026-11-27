"""Combine distinct normal-entry rows executed with identical frozen source."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();states=[json.loads(f.read_text())for f in a.states]
    first=states[0];rows=[];seen=set()
    for state in states:
        assert state['source_commit']==first['source_commit']
        assert state['files']==first['files'],'source/config fingerprint set changed'
        for row in state['rows']:
            identity=(row['hand'],row['case'])
            assert identity not in seen,'duplicate case cannot be selected by outcome'
            assert row['status']=='terminal'
            rows.append(row);seen.add(identity)
    assert len(rows)==10
    combined=dict(source_commit=first['source_commit'],files=first['files'],rows=rows,completed=True,
                  component_states=[str(f.resolve())for f in a.states],
                  selection_rule='all ten pre-existing fixtures, one terminal row each; identical source/config bytes')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(combined,indent=2)+'\n')
    print('Ten distinct terminal cases share one frozen source manifest')
if __name__=='__main__':main()
