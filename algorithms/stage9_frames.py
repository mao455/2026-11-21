"""Nonblocking latest valid exposure buffer; no private state or reconstruction."""
class S9FrameInbox:
    def __init__(self,max_age_s=.1):
        self.latest=None;self.max_age_s=max_age_s
        self.counts=dict(new=0,duplicate=0,late=0,missing=0,future=0,stale_stop=0)
    def select(self,observation):
        o=dict(observation);c=o.get('image_state');frames=o.get('fingereye')
        valid=c is not None and frames and len({v.get('frame_id') for v in frames.values()})==1
        if not valid:self.counts['missing']+=1
        elif c['step']>o['step']:self.counts['future']+=1
        elif self.latest is not None and c['step']<self.latest[0]['step']:self.counts['late']+=1
        elif self.latest is not None and c['step']==self.latest[0]['step']:self.counts['duplicate']+=1
        else:
            if any(f['capture_step']!=c['step'] for f in frames.values()):raise ValueError('unpaired image state')
            self.latest=(c,frames);self.counts['new']+=1
        if self.latest is None or o['time_s']-self.latest[0]['time_s']>self.max_age_s+1e-10:
            self.counts['stale_stop']+=1;raise RuntimeError('no valid image within maximum age; safe stop')
        o['image_state'],o['fingereye']=self.latest
        return o
