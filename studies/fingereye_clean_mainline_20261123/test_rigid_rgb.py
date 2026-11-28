"""Rigid RGB update must preserve occluded shape yet react to face rotation."""
import unittest
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from mainline import strategy as m

class Volume:
    def __init__(self,points,yaw,partial=False):
        rot=Rotation.from_euler('z',yaw).as_matrix();p=points@rot.T
        self.data={}
        for name in ['a','b']:
            mask=np.zeros((140,140),np.uint8)
            uv=np.rint(p[:,:2]*4000+70).astype(np.int32)
            cv2.fillConvexPoly(mask,cv2.convexHull(uv),1)
            if partial:mask[:,:70]=0
            mask=mask.astype(bool)
            self.data[name]=dict(mask=mask,camera=np.eye(4),robot_depth=np.ones(mask.shape)*10.,
                sdf=distance_transform_edt(~mask)-distance_transform_edt(mask))
    def frames(self,obs):return {},self.data
    def project(self,name,points,camera):
        uv=points[:,:2]*4000+70
        valid=(uv>=0).all(1)&(uv<139).all(1)
        return uv,np.ones(len(points)),valid

class RigidRGB(unittest.TestCase):
    def setUp(self):
        self.cloud=np.array([[x,y,z]for x in [-.012,.012]for y in [-.003,.003]for z in [0.,.02]])
        self.info=dict(collision_points=self.cloud,center=np.array([0.,0.,.01]),axes=np.eye(2),half=np.array([.012,.003,.01]),table_z=0.,top_z=.02)
        self.obs=dict(step=4,time_s=4/120,image_state=dict(step=4,time_s=4/120,actual_q=[],actual_qvel=[]),fingereye=dict(a=dict(frame_id=1)))
    def test_partial_mask_does_not_shrink_or_move_shape(self):
        new,report=m.register_rgb_rigid(Volume(self.cloud,0.,True),self.obs,self.info)
        self.assertTrue(report['valid']);self.assertTrue(report['pose_ambiguous_within_envelope'])
        np.testing.assert_array_equal(new['collision_points'],self.cloud)
    def test_admissible_prior_is_not_moved_to_chase_small_outlier(self):
        volume=Volume(self.cloud,0.)
        for f in volume.data.values():
            f['mask'][30:50,120:130]=True
            f['sdf']=distance_transform_edt(~f['mask'])-distance_transform_edt(f['mask'])
        new,report=m.register_rgb_rigid(volume,self.obs,self.info)
        fractions=[v['explained_fraction']for v in report['views_before'].values()]
        self.assertTrue(all(.9<=v<.95 for v in fractions))
        self.assertTrue(report['valid']);self.assertTrue(report['pose_ambiguous_within_envelope'])
        np.testing.assert_array_equal(new['collision_points'],self.cloud)
    def test_recovery_keeps_same_observed_shape(self):
        volume=Volume(self.cloud,0.,True)
        new,report=m.stage3_live_geometry(volume,self.obs,self.info)
        self.assertTrue(report['valid']);self.assertTrue(report['shape_preserved'])
        self.assertIs(volume.info,new)
        np.testing.assert_array_equal(new['collision_points'],self.cloud)
        np.testing.assert_array_equal(new['half'],self.info['half'])
    def test_calibrated_continuous_ray_coverage_has_no_sample_holes(self):
        from types import SimpleNamespace
        calibration=dict(projection_type='pinhole',fx=220.,fy=220.,cx=70.,cy=70.)
        volume=SimpleNamespace(calibration=dict(a=calibration),size=(140,140))
        camera=np.eye(4);camera[2,3]=.12
        frame=dict(camera=camera)
        queries=np.array([[70.,70.],[90.,70.],[110.,70.]])
        distances=m._rgb_envelope_ray_distances(volume,'a',frame,self.cloud,queries)
        np.testing.assert_allclose(distances[:2],0.,atol=1e-8)
        self.assertGreater(distances[2],10.)

    def test_local_solver_can_still_move_after_550_cpu_seconds(self):
        from unittest.mock import patch
        with patch.object(m.time,'process_time',return_value=551.):
            fit=m.least_squares(lambda x:x-2.,np.array([0.]),max_nfev=20)
        self.assertLess(abs(fit.x[0]-2.),1e-6)

    def test_fixed_center_rotation_changes_surface_pose(self):
        new,report=m.register_rgb_rigid(Volume(self.cloud,.35),self.obs,self.info)
        self.assertTrue(report['valid']);self.assertFalse(report['pose_ambiguous_within_envelope'])
        self.assertGreater(abs(report['parameters'][2]),.25)
        self.assertLess(np.linalg.norm(new['center'][:2]),.001)
        self.assertGreater(np.linalg.norm(new['collision_points']-self.cloud,axis=1).max(),.002)
        self.rotation_report=report

if __name__=='__main__':unittest.main()
