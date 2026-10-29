import glob, statistics as st
from pathlib import Path
from explicometro import io as aio
n=0; s=0.0; mx=0.0; gaps=[]
for f in sorted(glob.glob("data/interim/es_congreso_202_q*/diarization.rttm")):
    segs=sorted(aio.read_rttm(Path(f)), key=lambda x:x.start)
    for i,a in enumerate(segs):
        for b in segs[i+1:]:
            if b.speaker==a.speaker: continue
            o=min(a.end,b.end)-max(a.start,b.start)
            if o>0: n+=1; s+=o; mx=max(mx,o)
    gaps+=[b.start-a.end for a,b in zip(segs,segs[1:]) if a.speaker!=b.speaker]
print("solapes>0:",n,"total_s",round(s,2),"max_s",round(mx,2))
print("cambios de hablante:",len(gaps),"hueco mediano s",round(st.median(gaps),2),"min",round(min(gaps),2))
