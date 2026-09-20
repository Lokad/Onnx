"""Fixed balanced batches, independent runtimes, and unchanged timing screens."""
from itertools import permutations
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
POLICIES=['default','memory'];ROLES=['A','B','C','N']
ORDERS=[['A','B','N','C'],['B','C','A','N'],['C','N','B','A'],['N','A','C','B']]
PROTOCOL='interleaved-independent-processes-v1'
CORE='8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1'
NATIVE='13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b'
LIMITS=dict(seconds=600,rss=12*1024**3,available=1024**3)
CRITERIA=dict(control_aggregate=[.995,1.005],control_visit=[.99,1.01],control_position=[.99,1.01],
    control_position_contrast=1.01,solo_bridge=[.95,1.05],candidate_aggregate=[.98,.99,1.01,1.01,1.01],
    candidate_visit=1.02,native_scaled_error=1e-4,foreign_cpu=.02,steal=.005)

def schedule():
    jobs=[]
    for visit in range(4):
        for index in (range(5) if visit%2==0 else reversed(range(5))):
            for policy_index in ([0,1] if (visit+index)%2==0 else [1,0]):
                jobs.append(dict(name=f'v{visit}-{CASES[index]}-{POLICIES[policy_index]}',case=CASES[index],case_index=index,
                    visit=visit,policy=POLICIES[policy_index],policy_index=policy_index,creation=ORDERS[(visit+index+policy_index)%4]))
    return jobs

def cycles(job,smoke=False):
    state=20260920+100*job['visit']+10*job['case_index']+job['policy_index'];rows=[]
    for block in range(2):
        values=list(permutations(ROLES))
        for i in range(len(values)-1,0,-1):
            state=(state*1664525+1013904223)&0xffffffff;j=state%(i+1);values[i],values[j]=values[j],values[i]
        rows.extend([list(v) for v in values])
    return rows[:2] if smoke else rows

def commands(job,smoke=False):
    result=[]
    for i,role in enumerate(job['creation']):
        result.extend([(role,0,-1),(role,1,-1)])
        if i==0:result.append((role,2,0))
    for cycle,order in enumerate(cycles(job,smoke)):
        result.extend((role,3,cycle) for role in order)
    result.extend((role,4,-1) for role in job['creation'][:-1])
    result.extend([(job['creation'][-1],2,1),(job['creation'][-1],4,-1)])
    return result

def specification(job,phase,smoke=False):
    return dict(protocol=PROTOCOL,phase=phase,policy=job['policy'],role=job['role'],visit=job['visit'],case_index=job['case_index'],
        blocks=2 if smoke else 48,calls=2 if smoke else [32,16,4,4,2][job['case_index']],conditioning_seconds=.05 if smoke else 30,
        conditioning_minimum=2 if smoke else 128,solo_calls=2 if smoke else 64,smoke=smoke)
