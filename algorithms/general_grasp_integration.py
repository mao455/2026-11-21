"""Legal RGB-to-multifinger proposals. Never reads private scene/evaluation data.

Pure silhouettes remain search hypotheses. Bounded local correspondences may
enter the generic planner, with provenance and conditional uncertainty intact.
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from general_grasp_contacts import URDFHand, ContactPlanner
from general_grasp_vision import (estimate_cross_section, silhouette_cross_section,
                                  sift_correspondence, correspondence_patches)


class InitialRGBConstraint:
    """One immutable initial silhouette, conditional on precontact stationarity."""
    def __init__(self, native):
        self.native = native

    def project(self, name, points, frame):
        if name != 'initial_overhead':
            return self.native.project(name, points, frame)
        local = (np.asarray(points)-frame['camera'][:3,3])@frame['camera'][:3,:3]
        k=frame['intrinsics']
        uv=local[:,:2]/np.maximum(local[:,2:3],1e-12)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]-.5
        valid=(local[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<=frame['size'][0]-1)&(uv[:,1]>=0)&(uv[:,1]<=frame['size'][1]-1)
        return uv,np.linalg.norm(local,axis=1),valid


def precontact_cross_section(native, observation, center, axis, initial_q, search_halfwidth_m):
    """Intersect initial RGB and current FingerEye constraints; no new image.

    The initial pose is the robot's initial actual q, not an object pose. The
    initial frame must not constrain an object after closure or displacement.
    All surface targets remain inferred, with unknown inward depth preserved.
    """
    key=(observation['step'],tuple(np.asarray(initial_q)))
    cached=getattr(native,'_precontact_constraints',None)
    if cached is not None and cached[0]==key:
        frames,correspondence=cached[1:]
    else:
        _,frames=native.frames(observation)
        correspondence=sift_correspondence(native,frames)
    local=correspondence_patches(correspondence,center,axis)
    top=observation['initial_overhead'];rgb=np.load(top['rgb_path'],allow_pickle=False)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);delta=abs(hsv[:,:,0].astype(float)-native.hue)
    mask=(np.minimum(delta,180-delta)<12)&(hsv[:,:,1]>60)&(hsv[:,:,2]>35)
    pose=np.asarray(top['camera_pose_world']);rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    camera=np.eye(4);camera[:3,:3]=rotation;camera[:3,3]=pose[:3]
    k=np.asarray(top['intrinsics']);yy,xx=np.indices(rgb.shape[:2])
    rays=np.c_[((xx.ravel()+.5)-k[0,2])/k[0,0],((yy.ravel()+.5)-k[1,2])/k[1,1],np.ones(xx.size)]
    rays/=np.linalg.norm(rays,axis=1)[:,None]
    # RobotVisibility transforms arbitrary local rays by the supplied pose.
    if 'initial_overhead' not in frames:
        depth=native.visibility.depth(native.robot.fk(np.asarray(initial_q)),{'initial_overhead':(camera,rays)})['initial_overhead']
        frames['initial_overhead']=dict(size=(rgb.shape[1],rgb.shape[0]),camera=camera,intrinsics=k,
            mask=mask,sdf=distance_transform_edt(~mask)-distance_transform_edt(mask),
            usable=bool(mask.sum()>=5),frame_id=0,robot_depth=depth.reshape(rgb.shape[:2]))
        native._precontact_constraints=(key,frames,correspondence)
    silhouette=silhouette_cross_section(InitialRGBConstraint(native),frames,center,axis,
        search_halfwidth_m=search_halfwidth_m,spacing_m=.00025,patch_radius_m=.001,pixel_margin=1.)
    patches=[a if a['point'] is not None else b for a,b in zip(local,silhouette['patches'])]
    return dict(patches=patches,silhouette=silhouette,correspondence=correspondence,
                consumed_camera_names=list(observation['fingereye']),
                initial_rgb_used=True,stationarity_assumed=True,strict_success_claim=False)


def generic_local_proposal(robot_description, actual_q, geometry, observation, info,
                           safety_check=None, initial_q=None, contact_counts=None, finger_groups=None):
    hand = URDFHand.from_urdf(robot_description['urdf_path'],
                              joint_order=robot_description['joint_names'])
    planner = ContactPlanner(hand, max_evaluations=45, beam_width=8,
                             anchors_per_pair=1, candidates_per_count=2)
    patches, sections = [], []
    for direction in (np.r_[info['axes'][:,0],0.], np.r_[info['axes'][:,1],0.]):
        if initial_q is None:
            estimate = estimate_cross_section(geometry, observation, info['center'], direction,
                                              search_halfwidth_m=float(max(info['half'][:2])+.01))
        else:
            estimate = precontact_cross_section(geometry, observation, info['center'], direction,
                                                initial_q,float(max(info['half'][:2])+.01))
        patches.extend(estimate['patches'])
        sections.append(dict(patches=estimate['patches'],
                             cameras=estimate['consumed_camera_names']))
    try:
        proposal = planner.plan(np.asarray(actual_q), patches, safety_check=safety_check,
                                contact_counts=contact_counts, finger_groups=finger_groups).as_dict()
    except ValueError as error:
        # Native physics can report a joint slightly outside the authored bound.
        # Keep the actual measurement; do not clip it into a fabricated state.
        proposal = dict(feasible=False, q=None, contacts=[],
                        diagnostics=dict(reason=('invalid_measured_robot_state' if 'Actual q' in str(error)
                                                 else 'invalid_planner_request'), detail=str(error)))
    return dict(proposal=proposal, sections=sections,
                input_camera_names=list(observation['fingereye']),
                physical_execution=False,
                reason='Only a proposal; unchanged physical policy still owns the action path')
