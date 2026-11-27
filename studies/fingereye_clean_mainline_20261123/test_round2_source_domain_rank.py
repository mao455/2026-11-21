"""Asset-free geometry and shared ranking invariants for the production D69 helper."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

PATH = Path(__file__).parent/'mainline/contact_regions.py'
spec = importlib.util.spec_from_file_location('d69_under_test', PATH)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


def rectangle(x0, x1, y0, y1):
    return np.array([[x0,y0,0.],[x1,y0,0.],[x1,y1,0.],[x0,y1,0.]])


class ToyRobot:
    """Named joints and independent fixed source regions; no asset fixtures."""
    def __init__(self, count, reverse=False):
        self.finger_names = tuple('f'+str(i) for i in range(count))
        canonical = ['base_x','base_y','base_z','yaw','pitch','roll']+list(self.finger_names)
        self.names = tuple(reversed(canonical)) if reverse else tuple(canonical)
        self.description = SimpleNamespace(index={n:i for i,n in enumerate(self.names)}, pads=('original',))
        self.base_indices = np.array([self.description.index[n] for n in canonical[:6]])
        self.all_finger_indices = np.array([self.description.index[n] for n in self.finger_names])
        self.active_finger_indices = self.all_finger_indices.copy()
        self.lo = np.full(len(self.names),-1.); self.hi = np.full(len(self.names),1.)
        self.upright_reference = np.eye(3)
        self.select_pair(self.finger_names, [0]*count)
    def select_pair(self, group, regions=None):
        self.finger_names = tuple(group)
        if regions is None: regions=[0]*len(group)
        self.contact_region_indices = tuple(regions)
        self.contact_links = tuple('link_'+f for f in group)
        # Region0 deliberately has a worse geometric score than1/2, allowing
        # the task/table priority tests to detect accidental class reversal.
        self.contact_points = np.array([[.005 if r==0 else .002,0.,0.] for r in regions])
        self.contact_normals = np.tile([1.,0.,0.],(len(group),1))
        self.selected_patches = [dict(finger=f,link=l,surface_link=l,
            point_m=p.tolist(),outward_normal=[1.,0.,0.],support_radius_m=.0012)
            for f,l,p in zip(group,self.contact_links,self.contact_points)]
        self.description.pads = tuple((f,r) for f,r in zip(group,regions))
    def fk(self, q):
        return {link:np.eye(4) for link in self.contact_links}
    def tips(self, q):
        return self.contact_points.copy(),self.contact_normals.copy(),None,self.fk(q)


class FixedDomainAPI:
    @staticmethod
    def generate_contact_sets(info, count, **kwargs):
        # outward=-x and +.3mm stand-off give the target x=.002 exactly.
        return [dict(points=np.tile([.0023,0.,0.],(count,1)),
                     outward_normals=np.tile([-1.,0.,0.],(count,1)))]
    @staticmethod
    def ordered_surface_planes(goals, inward):
        return tuple(tuple(np.r_[n,np.dot(n,p)].round(9)) for p,n in zip(goals,-inward))
    @staticmethod
    def planar_contact_cells(robot, patch):
        return dict(status='fixed_center',cells_local_m=[],support_radius_m=patch['support_radius_m'])


def unchanged_fit(robot, q, targets, inward, fraction=None, normal_weight=12.):
    return np.asarray(q).copy()


def records_for(robot, only_one=False, seed=0):
    count=len(robot.finger_names); targets=np.tile([.002,0.,0.],(count,1)); inward=np.tile([1.,0.,0.],(count,1))
    planes=FixedDomainAPI.ordered_surface_planes(targets,inward)
    rows=[]
    for region in ([0] if only_one else [0,1,2]):
        rows.append(dict(regions=[region]*count,score=20.25 if region==0 else 0.,seed_index=seed,
            target_set=0,permutation=list(range(count)),ordered_outward_planes=[list(p) for p in planes],
            table_priority_proven_impossible=region==2,
            nominal_surface_task=dict(task_feasible=region!=1)))
    return rows


def run_rank(*args, **kwargs):
    # Exercise the production ranking function. Only source/target fixtures are
    # replaced, so tests require neither meshes nor IsaacLab.
    with patch.multiple(helper,
            generate_contact_sets=FixedDomainAPI.generate_contact_sets,
            planar_contact_cells=FixedDomainAPI.planar_contact_cells,
            ordered_surface_planes=FixedDomainAPI.ordered_surface_planes):
        return helper.source_domain_corrected_rank(*args, **kwargs)


class SourceDomainRankTests(unittest.TestCase):
    def test_non_coplanar_query_projects_to_finite_polygon(self):
        cell=helper._compile_source_rank_cell(rectangle(-1,1,-1,1))
        np.testing.assert_allclose(helper._closest_source_rank_cell(cell,np.array([.2,.3,8.])),[.2,.3,0.],atol=1e-12)
        np.testing.assert_allclose(helper._closest_source_rank_cell(cell,np.array([3.,.3,8.])),[1.,.3,0.],atol=1e-12)

    def test_disconnected_union_and_hole_are_not_bridged(self):
        query=np.array([0.,0.,2.])
        disconnected=[rectangle(-3,-1,-1,1),rectangle(1,3,-1,1)]
        ring=[rectangle(-2,-1,-2,2),rectangle(1,2,-2,2),rectangle(-1,1,-2,-1),rectangle(-1,1,1,2)]
        for cells in (disconnected,ring):
            points=np.array([helper._closest_source_rank_cell(helper._compile_source_rank_cell(c),query) for c in cells])
            nearest=points[np.argmin(np.sum((points-query)**2,axis=1))]
            self.assertAlmostEqual(np.linalg.norm(nearest[:2]),1.,places=12)
            self.assertFalse(np.allclose(nearest[:2],query[:2]))

    def test_closest_point_is_yaw_equivariant(self):
        cell=rectangle(-2,1,-1,1); query=np.array([2.,.2,3.])
        R=Rotation.from_euler('z',.73).as_matrix()
        actual=helper._closest_source_rank_cell(helper._compile_source_rank_cell(cell@R.T),query@R.T)
        expected=helper._closest_source_rank_cell(helper._compile_source_rank_cell(cell),query)@R.T
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_proxy_fixed_fallback_and_priority_classes(self):
        robot=ToyRobot(3); before_points=robot.contact_points.copy(); before_pads=robot.description.pads
        before_regions=robot.contact_region_indices; original=records_for(robot)
        rows,report=run_rank(robot,np.zeros(len(robot.names)),{},original,[],(),
            grasp_seed=unchanged_fit)
        self.assertEqual([r['regions'] for r in rows],[[0]*3,[1]*3,[2]*3])
        self.assertEqual(len(rows),len(original)); self.assertEqual(report['new_optimizer_calls'],0)
        self.assertIs(robot.description.pads,before_pads)
        self.assertEqual(robot.contact_region_indices,before_regions)
        np.testing.assert_array_equal(robot.contact_points,before_points)
        for row in rows:
            self.assertEqual(row['nominal_surface_task'],original[row['source_domain_original_rank']]['nominal_surface_task'])
            for source in row['source_domain_one_step']['sources']:
                self.assertEqual(source['status'],'fixed_center'); self.assertIsNone(source['chosen_cell'])
                self.assertEqual(source['center_shift_m'],0.)
                self.assertEqual(source['unchanged_radius_m'],.0012)

    def test_two_three_normalization_and_named_control_order(self):
        scores=[]; by_name=[]
        for count,reverse in [(2,False),(3,False),(3,True)]:
            robot=ToyRobot(count,reverse); prior=np.zeros(len(robot.names))
            rows,report=run_rank(robot,prior,{},records_for(robot,True,116),[],(),
                grasp_seed=unchanged_fit)
            self.assertLess(report['original_score_reproduction_max_error'],1e-12)
            scores.append(rows[0]['score'])
            q=rows[0]['source_domain_one_step']['q_initializer']
            by_name.append(dict(zip(robot.names,q)))
        np.testing.assert_allclose(scores,[20.25]*3,atol=1e-12)
        self.assertEqual(by_name[1],by_name[2])


if __name__=='__main__':
    unittest.main(verbosity=2)
