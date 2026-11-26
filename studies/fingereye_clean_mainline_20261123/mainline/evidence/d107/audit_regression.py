"""Offline reproducibility checks; never imported by the online policy."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commands(episode):
    result = []
    for line in (episode / 'agent_trace.jsonl').open():
        row = json.loads(line)
        if row.get('op') == 'step':
            result.append({k: row[k] for k in ('target_q', 'steps', 'phase')})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.runs / 'STATE.json').read_text())
    assert all(row['status'] == 'terminal' for row in state['rows'])
    changed = [path for path, value in state['files'].items()
               if digest(Path(path)) != value]
    assert not changed, changed
    rows = []
    for item in state['rows']:
        session = Path(item['session'])
        old = args.baseline / session.name
        new_commands = commands(session / 'episode')
        old_commands = commands(old / 'episode')
        mismatches = [i for i, pair in enumerate(zip(old_commands, new_commands))
                      if pair[0] != pair[1]]
        receipt = json.loads((session / 'budget_receipt.json').read_text())
        result = json.loads((session / 'episode/result.json').read_text())
        before = json.loads((old / 'episode/result.json').read_text())
        assert result['controller_sha256'] == state['controller_sha256']
        assert result['evaluator_and_adapter_unchanged']
        assert not receipt['leftover_descendants']
        assert not receipt['timeout_triggered']
        assert receipt['source_unchanged']
        rows.append(dict(
            case=item['case'], hand=item['hand'],
            old_session=str(old), new_session=str(session),
            baseline_success=before['success'], success=result['success'],
            old_commands=len(old_commands), new_commands=len(new_commands),
            identical_commands=(not mismatches and len(old_commands) == len(new_commands)),
            first_command_mismatch=mismatches[0] if mismatches else None,
            source_sha_unchanged=True,
            evaluator_and_adapter_unchanged=True,
            leftover_descendants=receipt['leftover_descendants'],
            actual_host_elapsed_s=receipt['actual_host_elapsed_s']))
    args.output.write_text(json.dumps(dict(
        source_commit=state['source_commit'],
        controller_sha256=state['controller_sha256'],
        all_pinned_source_files_unchanged=True,
        rows=rows), indent=2))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
