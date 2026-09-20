"""Fixed independent-process schedule and immutable empirical decision limits."""
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
POLICIES=['default','memory']
ROLES=['A','B','C','N']
ORDERS=[['A','B','N','C'],['B','C','A','N'],['C','N','B','A'],['N','A','C','B']]
PROTOCOL='fingerprint-isolated-deployment-v1'
CORE='48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710'
NATIVE='13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b'
LIMITS=dict(seconds=300,rss=6*1024**3,available=1024**3)
CRITERIA=dict(control_aggregate=[.995,1.005],control_visit=[.99,1.01],control_position=[.99,1.01],
              control_position_contrast=1.01,candidate_aggregate=[.98,.99,1.01,1.01,1.01],candidate_visit=1.02,
              native_scaled_error=1e-4,foreign_cpu=.02,steal=.005)

def schedule():
    rows=[]
    for visit in range(4):
        for index in (range(5) if visit%2==0 else reversed(range(5))):
            for policy_index in ([0,1] if (visit+index)%2==0 else [1,0]):
                policy=POLICIES[policy_index]
                for position,role in enumerate(ORDERS[(visit+index+policy_index)%4]):
                    rows.append(dict(name=f'v{visit}-{CASES[index]}-{policy}-{role}',case=CASES[index],case_index=index,
                                     visit=visit,policy=policy,role=role,position=position))
    return rows

def specification(job,phase,smoke=False):
    assert phase in ['aa','compare']
    return dict(protocol=PROTOCOL,phase=phase,policy=job['policy'],role=job['role'],visit=job['visit'],case_index=job['case_index'],
                blocks=2 if smoke else 48,calls=2 if smoke else [32,16,4,4,2][job['case_index']],conditioning_seconds=.1 if smoke else 30,smoke=smoke)
