"""Finite observed contact regions shared by IK and endpoint admission.

Only the public RGB envelope and static contact footprint enter this model.
A tangential offset inside a face is a change of contact center, not a miss.
Normal separation and distance beyond a finite supported edge remain errors.
"""
import numpy as np
from scipy.spatial import ConvexHull
try:
    from contact_regions import finite_support_region
except ImportError:
    from mainline.contact_regions import finite_support_region


class ContactIntent:
    def __init__(self, robot, targets, inward, info=None):
        self.targets=np.asarray(targets,float)
        self.inward=np.asarray(inward,float)
        self.regions=[]
        self.normals=[]
        self.height_uncertainty_m=float((info or {}).get("position_uncertainty_m",.0015))
        cloud=None if info is None else np.asarray(info['collision_points'],float)
        hull=None
        if cloud is not None:
            cached=getattr(robot,'_contact_intent_hull_cache',None)
            if cached is not None and cached[0] is info['collision_points']:hull=cached[1]
            else:
                hull=ConvexHull(cloud)
                robot._contact_intent_hull_cache=(info['collision_points'],hull)
            cloud=cloud[hull.vertices]
        for target,normal,patch in zip(self.targets,-self.inward,robot.selected_patches):
            region=None
            if hull is not None:
                plane=hull.equations[np.argmax(hull.equations[:,:3]@normal)]
                vertices=np.unique(np.round(cloud[abs(cloud@plane[:3]+plane[3])<2e-7],10),axis=0)
                if len(vertices)>=3:
                    try:
                        # Edge and partial-footprint contacts are physically valid.
                        # Static footprint size belongs to quality/load scoring;
                        # eroding the observed face would forbid those contacts.
                        # Stand-off is on the same observed facet used by surface IK.
                        vertices=vertices+normal*.0003
                        origin=vertices.mean(0)
                        if len(vertices)>2:
                            _,_,basis=np.linalg.svd(vertices-origin,full_matrices=False)
                            basis=basis[:2];xy=(vertices-origin)@basis.T
                            boundary=ConvexHull(xy)
                            region=(origin,basis,xy[boundary.vertices],boundary.equations)
                    except (ValueError,RuntimeError):pass
            self.regions.append(region)
            self.normals.append(normal if region is None else plane[:3].copy())

    def goals(self, points):
        output=[]
        for point,target,region in zip(points,self.targets,self.regions):
            if region is None:output.append(target);continue
            origin,basis,polygon,planes=region
            xy=(point-origin)@basis.T
            if np.max(planes[:,:2]@xy+planes[:,2])>0:
                a=polygon;b=np.roll(polygon,-1,axis=0);edge=b-a
                t=np.clip(np.sum((xy-a)*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
                candidates=a+t[:,None]*edge
                xy=candidates[np.argmin(np.sum((candidates-xy)**2,axis=1))]
            output.append(origin+xy@basis)
        return np.asarray(output)

    def height_error(self, points):
        # Proposal height is an observed-region hypothesis, not a free axis on
        # an incompletely observed tall extrusion. Other heights remain normal
        # candidate proposals. Horizontal motion on a wide face stays free.
        return np.maximum(abs(np.asarray(points)[:,2]-self.targets[:,2])-self.height_uncertainty_m,0.)

    def errors(self, points):
        return np.linalg.norm(np.asarray(points)-self.goals(points),axis=1)

    def residual(self, points):
        return (np.asarray(points)-self.goals(points)).ravel()*1500

    def report(self, points):
        goals=self.goals(points)
        offset=np.asarray(points)-goals
        normal=np.asarray(self.normals)
        signed=np.sum(offset*normal,axis=1)
        tangential=offset-signed[:,None]*normal
        return dict(targets=goals.tolist(),per_finger_error_m=self.errors(points).tolist(),
            normal_gap_m=signed.tolist(),tangential_outside_region_m=np.linalg.norm(tangential,axis=1).tolist(),
            selected_facet_normals=normal.tolist(),
            finite_region_contacts=[r is not None for r in self.regions],height_search_uncertainty_m=self.height_uncertainty_m,height_offset_diagnostic_m=self.height_error(points).tolist(),
            tangential_center_shift_m=np.linalg.norm(goals-self.targets,axis=1).tolist(),
            source='public RGB facet including edge contacts; nearest finite contact point plus 0.3mm stand-off; static footprint size is a load-quality assumption',
            normal_chord_weight=2.5,position_weight_per_m=1500.,position_limit_m=.0008,normal_limit_rad=.5)

    def support_radii(self, points, maximum):
        radii=[]
        for point,region,cap in zip(self.goals(points),self.regions,maximum):
            if region is None:radii.append(0.);continue
            origin,basis,polygon,planes=region
            distance=-(planes[:,:2]@((point-origin)@basis.T)+planes[:,2])
            radii.append(max(0.,min(float(cap),float(distance.min()))))
        return np.asarray(radii)
