"""Two process-lifetime leases, restricted to this experiment's simulators."""
import fcntl,json,os,time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def _slot(output,kind):
    started=time.perf_counter()
    while True:
        lease=None
        for number in range(2):
            candidate=open('/tmp/fingereye_stage14_'+kind+'_'+str(number)+'.lock','a+')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close();continue
            lease=candidate;break
        if lease is not None:break
        time.sleep(.2)
    try:
        record=dict(pid=os.getpid(),slot=number,wait_s=time.perf_counter()-started,output=str(output))
        lease.seek(0);lease.truncate();lease.write(json.dumps(record));lease.flush()
        Path(output).mkdir(parents=True,exist_ok=True)
        (Path(output)/('stage14_'+kind+'_lease.json')).write_text(json.dumps(record,indent=2))
        yield record
    finally:
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

@contextmanager
def simulator_slot(output):
    with _slot(output,'simulator') as lease:yield lease

@contextmanager
def admission_slot(output):
    # Acquired before the host starts its backend initialization RPC deadline.
    # The backend keeps its separate lifetime lease as a second hard bound.
    with _slot(output,'admission') as lease:yield lease
