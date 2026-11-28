"""Whole-hand FK ranks regions; exact meshes admit only actual IK/path solutions."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


class KinematicCandidateBank:
    def __init__(self, robot, prior, checker):
        self.robot=robot;self.checker=checker;self.prior=np.asarray(prior,float).copy()
        self.last_note=time.monotonic();self.rank_count=0
        active=sorted(robot.all_finger_indices,key=lambda i:robot.names[i])
        self.configs=[self.prior.copy()]
        for fraction in (.15,.4,.7):
            q=self.prior.copy();q[active]=robot.lo[active]+fraction*(robot.hi[active]-robot.lo[active]);self.configs.append(q)
        for values in qmc.Sobol(len(active),scramble=False).random_base2(6):
            q=self.prior.copy();q[active]=robot.lo[active]+values*(robot.hi[active]-robot.lo[active]);self.configs.append(q)
        self.poses=[];self.minimum_z=[];self.self_penalty=[]
        for q in self.configs:
            self.normalize(q);poses=robot.fk(q);self.poses.append(poses)
            low,penalty=self.geometry(poses);self.minimum_z.append(low);self.self_penalty.append(penalty)
            self.heartbeat('whole_hand_FK_seed_bank')
        self.cache={};self.groups={};self.table_z=checker.table_z

    def normalize(self,q):
        r=self.robot;q[:]=np.clip(q,r.lo+1e-7,r.hi-1e-7);q[r.base_indices]=0.

    def heartbeat(self,stage):
        now=time.monotonic()
        if now-self.last_note>=15. and hasattr(self.robot,'planning_note'):
            self.last_note=now
            self.robot.planning_note('planning_progress',dict(stage=stage,process_cpu_s=time.process_time(),observations_requested=False,physics_steps_requested=0))

    def geometry(self,poses):
        r=self.robot;c=self.checker
        low=float((r.vertices(poses)@r.upright_reference.T)[:,2].min())
        penalty=0.
        # Coarse separation is a seed score, never a collision certificate.
        # Full FCL on every random configuration previously consumed the CPU
        # budget before an actual contact candidate had even been optimized.
        if hasattr(c,'hulls') and len(c.pairs):
            bounds=[]
            for name in c.names:
                T=poses[name];vertices=c.hulls[name]@T[:3,:3].T+T[:3,3]
                bounds.append([vertices.min(0),vertices.max(0)])
            bounds=np.asarray(bounds);a,b=c.pairs[:,0],c.pairs[:,1]
            overlap=np.minimum(bounds[a,1],bounds[b,1])-np.maximum(bounds[a,0],bounds[b,0])
            depth=np.maximum(np.min(overlap,axis=1),0.).max(initial=0.)
            penalty=.5*(float(depth)*1000)**2
        return low,penalty

    def group_bank(self):
        r=self.robot;key=tuple(r.finger_names)
        if key not in self.groups:
            configs=list(self.configs);poses=list(self.poses);lows=list(self.minimum_z);penalties=list(self.self_penalty)
            # Keep the measured unused digits whenever they already leave a
            # route. Whole-hand alternatives remain available if parking helps.
            for fraction in (.15,.4,.7):
                q=self.prior.copy();ids=r.active_finger_indices
                q[ids]=r.lo[ids]+fraction*(r.hi[ids]-r.lo[ids]);self.normalize(q)
                fk=r.fk(q);low,penalty=self.geometry(fk)
                configs.append(q);poses.append(fk);lows.append(low);penalties.append(penalty)
            idle=r.unused_finger_indices
            if len(idle):
                motion=.5*np.mean(((np.asarray(configs)[:,idle]-self.prior[idle])/(r.hi[idle]-r.lo[idle]))**2,axis=1)
            else:motion=np.zeros(len(configs))
            self.groups[key]=(configs,poses,np.asarray(lows),np.asarray(penalties)+motion)
        return self.groups[key]

    def rank(self, targets, inward):
        r=self.robot;targets=np.asarray(targets);inward=np.asarray(inward)
        configs,poses,minimum_z,self_penalty=self.group_bank()
        key=(tuple(r.finger_names),tuple(r.contact_links),r.contact_points.tobytes(),r.contact_normals.tobytes())
        if key not in self.cache:
            p=np.array([[fk[link][:3,:3]@point+fk[link][:3,3]for link,point in zip(r.contact_links,r.contact_points)]for fk in poses])@r.upright_reference.T
            n=np.array([[fk[link][:3,:3]@normal for link,normal in zip(r.contact_links,r.contact_normals)]for fk in poses])@r.upright_reference.T
            self.cache[key]=(p,n)
        p,n=self.cache[key];center=p.mean(1);source=p-center[:,None,:];destination=targets-targets.mean(0)
        cosine=1500.**2*np.sum(source[:,:,:2]*destination[None,:,:2],axis=(1,2))+2.5**2*np.sum(n[:,:,:2]*inward[None,:,:2],axis=(1,2))
        sine=1500.**2*np.sum(source[:,:,0]*destination[None,:,1]-source[:,:,1]*destination[None,:,0],axis=1)+2.5**2*np.sum(n[:,:,0]*inward[None,:,1]-n[:,:,1]*inward[None,:,0],axis=1)
        theta=np.arctan2(sine,cosine);c=np.cos(theta);s=np.sin(theta);pp=p.copy();nn=n.copy()
        for a,b in [(pp,p),(nn,n)]:
            a[:,:,0]=c[:,None]*b[:,:,0]-s[:,None]*b[:,:,1]
            a[:,:,1]=s[:,None]*b[:,:,0]+c[:,None]*b[:,:,1]
        translation=targets.mean(0)-pp.mean(1);pp+=translation[:,None,:]
        position=np.sum((pp-targets)**2,axis=(1,2))*1500.**2
        normal=np.sum((nn-inward)**2,axis=(1,2))*2.5**2
        clearance=minimum_z+translation[:,2]-self.table_z
        cost=(position+normal)/len(targets)+.5*(np.minimum(clearance-.001,0.)*5000)**2+self_penalty
        hints=[];selected=np.argsort(cost,kind='stable')[:3]
        for index in selected:
            q=configs[index].copy();rotation=Rotation.from_euler('z',theta[index]).as_matrix()@r.upright_reference
            q[r.base_indices[:3]]=translation[index];q[r.base_indices[3:]]=Rotation.from_matrix(rotation).as_euler('ZYX')
            hints.append(np.clip(q,r.lo+1e-7,r.hi-1e-7))
        self.rank_count+=1
        if self.rank_count%128==0:self.heartbeat('whole_hand_region_proposals')
        return float(cost[selected[0]]),hints



def joint_region_proposals(bank,targets,inward,limit=8):
 r=bank.robot;targets=np.asarray(targets);inward=np.asarray(inward)
 if len(targets)!=2:return []
 configs,poses,_,penalties=bank.group_bank();result=[]
 regions=[np.asarray([x['point']for x in rr])for rr in r.patch_options];normals=[np.asarray([x['normal']for x in rr])for rr in r.patch_options]
 ii,jj=np.meshgrid(np.arange(len(regions[0])),np.arange(len(regions[1])),indexing='ij');ids=np.c_[ii.ravel(),jj.ravel()];dest=targets-targets.mean(0)
 for k,fk in enumerate(poses):
  pp=[];nn=[]
  for link,p,n in zip(r.contact_links,regions,normals):
   T=fk[link];pp.append(p@T[:3,:3].T+T[:3,3]);nn.append(n@T[:3,:3].T)
  p=np.stack([pp[0][ids[:,0]],pp[1][ids[:,1]]],axis=1);n=np.stack([nn[0][ids[:,0]],nn[1][ids[:,1]]],axis=1)
  center=p.mean(1);source=p-center[:,None,:]
  H=np.einsum('nki,kj->nij',source,dest)*1500**2+np.einsum('nki,kj->nij',n,inward)*12**2
  u,_,vt=np.linalg.svd(H);v=vt.transpose(0,2,1);z=np.broadcast_to(np.eye(3),(len(p),3,3)).copy();z[:,2,2]=np.linalg.det(v@u.transpose(0,2,1));rot=v@z@u.transpose(0,2,1)
  world=np.einsum('nij,nkj->nki',rot,source)+targets.mean(0);wn=np.einsum('nij,nkj->nki',rot,n)
  normal=np.arccos(np.clip(np.sum(wn*inward,axis=2),-1,1));err=np.linalg.norm(world-targets,axis=2)
  source_axis=fk[r.palm_link][:3,:3]@r.palm_forward_axis;axis=np.einsum('nij,j->ni',rot,source_axis);tilt=np.arccos(np.clip(-axis[:,2],-1,1))
  costs=np.sum(err**2,axis=1)*1500**2/2+np.sum((wn-inward)**2,axis=(1,2))*12**2/2+penalties[k]+(np.maximum(tilt-np.pi/4+.01,0)*30)**2
  for j in np.argsort(costs,kind='stable')[:5]:
   if tilt[j]>np.pi/4-.005:continue
   T=rot[j];q=configs[k].copy();q[r.base_indices[:3]]=targets.mean(0)-T@center[j];q[r.base_indices[3:]]=Rotation.from_matrix(T).as_euler('ZYX');q=np.clip(q,r.lo+2e-6,r.hi-2e-6)
   fk2=r.fk(q);clear=float(r.vertices(fk2)[:,2].min()-bank.checker.table_z);cost=float(costs[j]+.5*(min(clear-.001,0)*5000)**2)
   result.append(dict(score=cost,regions=tuple(int(i)for i in ids[j]),q=q,normal_error_rad=normal[j].tolist(),position_error_m=err[j].tolist(),table_clearance_m=clear,source='joint source-region SVD with whole-hand measured/Sobol articulation; exact endpoint/path still required'))
 result.sort(key=lambda x:x['score']);chosen=[];seen=set()
 for row in result:
  if row['regions']in seen:continue
  seen.add(row['regions']);chosen.append(row)
  if len(chosen)>=limit:break
 return chosen
