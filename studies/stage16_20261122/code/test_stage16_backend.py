"""Private evaluation placement must reproduce every actual-state verdict."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import stage16_backend as backend
HERE=Path(__file__).resolve().parent

def generated_backend_source():
    tree=ast.parse((HERE/'stage9_backend.py').read_text())
    main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    start=next(i for i,x in enumerate(main.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source' for t in x.targets))
    end=next(i for i,x in enumerate(main.body) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='mkdir')
    function=ast.FunctionDef(name='generate',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=main.body[start:end]+[ast.Return(value=ast.Name(id='source',ctx=ast.Load()))],decorator_list=[])
    ns={'HERE':HERE};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'generate','exec'),ns)
    return ns['generate']()

def test_generated_inline_and_deferred_sources_compile():
    source=generated_backend_source()
    for mode in ('inline','deferred'):compile(backend.transform(source,mode),mode,'exec')

def test_actual_snapshots_and_first_failure_preserved(tmp_path):
    class Audit:
        def check(self,bodies,obj):
            assert list(bodies)==['pad'];assert isinstance(obj,np.ndarray)
            return dict(passed=bool(obj[2]>.005),z=float(obj[2]))
    records=[dict(step=i,body_poses=[[0,0,0,1,0,0,0]],object_pose=[0,0,z,1,0,0,0],penetration=None) for i,z in enumerate((.01,.001,.02))]
    stream=io.StringIO();calls=[]
    def measure(name,fn,*args):calls.append(name);return fn(*args)
    backend.finalize_records(records,Audit(),['pad'],stream,tmp_path,measure)
    written=list(map(json.loads,stream.getvalue().splitlines()))
    assert written==records
    assert json.loads((tmp_path/'first_failure.json').read_text())==records[1]
    assert calls.count('offline_strict_geometry')==3


def test_failed_geometry_finalize_can_retry_without_duplicate_states(tmp_path):
    class Audit:
        failed = False
        def check(self, bodies, obj):
            if obj[0] == 1 and not self.failed:
                self.failed = True
                raise RuntimeError('transient geometry failure')
            return {'passed': bool(obj[0] != 1)}
    records=[dict(step=i,body_poses=[[0,0,0,1,0,0,0]],
        object_pose=[i,0,0,1,0,0,0],penetration=None) for i in range(3)]
    stream=io.StringIO();audit=Audit()
    def measure(name, fn, *args):return fn(*args)
    with pytest.raises(RuntimeError):
        backend.finalize_records(records,audit,['pad'],stream,tmp_path,measure)
    assert stream.getvalue()==''
    backend.finalize_records(records,audit,['pad'],stream,tmp_path,measure)
    assert list(map(json.loads,stream.getvalue().splitlines()))==records
    assert json.loads((tmp_path/'first_failure.json').read_text())==records[1]


def test_metadata_failure_does_not_repeat_successfully_committed_states(tmp_path):
    source=ast.parse(backend.transform(generated_backend_source(),'deferred'))
    finalize=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='_s16_finalize')
    # Execute the real generated closure in isolation, with globals standing in
    # for its enclosing backend locals. No physics or implementation copy.
    finalize.body=[ast.Global(names=n.names) if isinstance(n,ast.Nonlocal) else n for n in finalize.body]
    class Audit:
        def check(self,bodies,obj):return {'passed': True}
    records=[dict(step=i,body_poses=[[0,0,0,1,0,0,0]],
        object_pose=[i,0,0,1,0,0,0],penetration=None) for i in range(3)]
    stream=io.StringIO()
    def fail_metadata(*args):raise OSError('metadata write failed after trajectory commit')
    ns=dict(_s16_finalized=False,_marks={},_S16_MODE='deferred',records=records,
        audit=Audit(),env=SimpleNamespace(hand=SimpleNamespace(body_names=['pad'])),
        stream=stream,out=tmp_path,_measure=lambda name,fn,*args:fn(*args),
        _s16_finalize_records=backend.finalize_records,save=fail_metadata,time=backend.time)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[finalize],type_ignores=[])),'finalize','exec'),ns)
    with pytest.raises(OSError):ns['_s16_finalize']()
    ns['_s16_finalize']()
    assert list(map(json.loads,stream.getvalue().splitlines()))==records
    assert not (tmp_path/'first_failure.json').exists()


def test_partial_write_finalize_can_retry_without_truncated_or_duplicate_states(tmp_path):
    class Stream(io.StringIO):
        writes = 0
        def write(self, text):
            self.writes+=1
            if self.writes==2:
                super().write(text[:len(text)//2])
                raise OSError('transient partial write')
            return super().write(text)
    class Audit:
        def check(self, bodies, obj):return {'passed': True}
    records=[dict(step=i,body_poses=[[0,0,0,1,0,0,0]],
        object_pose=[i,0,0,1,0,0,0],penetration=None) for i in range(3)]
    stream=Stream()
    def measure(name, fn, *args):return fn(*args)
    with pytest.raises(OSError):
        backend.finalize_records(records,Audit(),['pad'],stream,tmp_path,measure)
    assert stream.getvalue()==''
    backend.finalize_records(records,Audit(),['pad'],stream,tmp_path,measure)
    assert list(map(json.loads,stream.getvalue().splitlines()))==records
