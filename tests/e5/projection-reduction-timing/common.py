"""Fixed prospective reduction-block comparison, standard library only."""
from pathlib import Path
import hashlib, itertools, json, math, random, statistics

ROOT = Path(__file__).resolve().parents[3]
CORE = '8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1'
PROBE = 'fc0049adfdc9ac6fa18db147721e14f53e4c3b74ae142dcb7d2835801e15ed55'
PROOF = '3f3619e680bbf870301e27f87969a0f7a83351e80cc45014e36e58af9cc5681b'
LIMITS = dict(seconds=600, rss=2*1024**3, available_memory=1024**3)
BANKS = [dict(name=n, rows=r, seed=s) for n,r,s in [('8',8,909),('30',30,931),('padded128',128,1029),('128',128,1030),('512',512,1413)]]
ORDERS = [[0,1,2,3,4],[4,3,2,1,0],[2,0,4,1,3],[3,1,4,0,2]]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path)
    with path.open('rb') as f: return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def write(path, value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)


def schedule():
    cycles=[]
    for worker in range(4):
        banks=[]
        for index in range(5):
            rng=random.Random(20260920+worker*17+index*101); bank=[]
            for _ in range(2):
                block=list(itertools.permutations(range(4))); rng.shuffle(block);bank.extend(map(list,block))
            banks.append(bank)
        cycles.append(banks)
    return dict(banks=BANKS,orders=ORDERS,cycles=cycles)


def validate_bank(bank, worker, index, frequency, sched):
    definition=BANKS[index]
    assert all(bank[k]==v for k,v in definition.items()) and bank['bank_index']==index and bank['worker']==worker
    assert bank['before']==bank['after']
    hashes=bank['before']; assert len(hashes['packed'])==72 and len(set(hashes['packed']))==72
    assert all(len(hashes[k])==64 for k in ['input384','input1536'])
    assert len(hashes['outputs'])==4 and len(hashes['outputs'][0])==72
    assert all(h==hashes['outputs'][0] for h in hashes['outputs'])
    assert all(len(v)==64 for values in hashes['outputs'] for v in values)
    permutations=sched['cycles'][worker][index]; warm=bank['conditioning']; measured=bank['measured']
    cycles=bank['conditioning_cycles'];assert 16<=cycles<=4096 and len(warm)==cycles*4 and len(measured)==192 and len(bank['first'])==4
    previous=0; allocated_total=0
    for phase,records in [('first',bank['first']),('conditioning',warm),('measured',measured)]:
        for i,row in enumerate(records):
            cycle=-1 if phase=='first' else i//4;position=i%4;mode=position if phase=='first' else permutations[cycle%48][position]
            assert (row['cycle'],row['position'],row['mode'])==(cycle,position,mode)
            assert all(isinstance(row[k],int) for k in ['start','end','allocated_thread','allocated_total','gc0','gc1','gc2'])
            assert 0<=previous<=row['start']<row['end'];previous=row['end']
            assert row['allocated_total']>=row['allocated_thread']>=0 and all(row[k]>=0 for k in ['gc0','gc1','gc2'])
            if phase=='measured':
                assert row['allocated_thread']==0 and row['gc0']==row['gc1']==row['gc2']==0
                allocated_total+=row['allocated_total']
    ticks=[sum(r['end']-r['start'] for r in warm if r['mode']==mode) for mode in range(4)]
    assert ticks==bank['conditioning_ticks'] and min(ticks)>=3*frequency
    # Stop at the first cycle satisfying BOTH workload and elapsed compute.
    if cycles>16:
        preceding=[sum(r['end']-r['start'] for r in warm[:-4] if r['mode']==mode) for mode in range(4)]
        assert min(preceding)<3*frequency
    return dict(means_ms=[statistics.mean((r['end']-r['start'])*1000/frequency for r in measured if r['mode']==mode) for mode in range(4)],
                conditioning_calls=len(warm),measured_calls=len(measured),total_allocated=allocated_total)


def verdict(workers):
    """Compute only fixed screens; a failure never drops observations."""
    result=[];control_ok=True;passes={2:True,3:True};primary={2:[],3:[]}
    for index,definition in enumerate(BANKS):
        means=[w['banks'][index]['means_ms'] for w in workers]
        controls=[m[1]/m[0] for m in means];aggregate=statistics.geometric_mean(controls)
        ordered=[]
        for before in [True,False]:
            modes=[[],[]]
            for worker in workers:
                raw=worker['raw'][index]['measured'];frequency=worker['identity']['frequency']
                for cycle in range(48):
                    rows=raw[cycle*4:cycle*4+4];by_mode={r['mode']:r for r in rows}
                    if (by_mode[0]['position']<by_mode[1]['position'])==before:
                        for mode in [0,1]:modes[mode].append((by_mode[mode]['end']-by_mode[mode]['start'])/frequency)
            ordered.append(statistics.mean(modes[1])/statistics.mean(modes[0]))
        contrast=ordered[0]/ordered[1]
        stable=.98<=aggregate<=1.02 and all(.95<=r<=1.05 for r in controls) and .95<=contrast<=1.05
        control_ok &= stable;candidates={}
        for mode in [2,3]:
            ratios=[[m[mode]/m[c] for c in [0,1]] for m in means]
            aggregates=[statistics.geometric_mean(r[c] for r in ratios) for c in [0,1]]
            is_primary=index in [1,2,3]
            gain=all(r<=.98 for r in aggregates) and all(r<=1.02 for pair in ratios for r in pair) if is_primary else all(r<=1.01 for r in aggregates)
            passes[mode] &= gain
            if is_primary: primary[mode].extend(aggregates)
            candidates[str(mode)]=dict(worker_ratios=ratios,aggregate_ratios=aggregates,screen_passed=gain)
        result.append(dict(name=definition['name'],means_ms=[statistics.mean(m[mode] for m in means) for mode in range(4)],
                           control=dict(workers=controls,aggregate=aggregate,order_ratios=ordered,order_contrast=contrast,passed=stable),candidates=candidates))
    qualified=[mode for mode in [2,3] if control_ok and passes[mode]]
    nominated=min(qualified,key=lambda mode:(statistics.geometric_mean(primary[mode]),-mode)) if qualified else None
    return dict(controls_passed=control_ok,candidate_screens={str(k):v for k,v in passes.items()},nominated=nominated,
                verdict='pass' if nominated else 'inconclusive' if not control_ok else 'rejected',banks=result)


def validate_resources(identity,samples):
    assert identity['complete'] and identity['code']==0 and not identity.get('error') and identity['limits']==LIMITS
    assert [r['name'] for r in identity['runs']]==[f'worker{i}' for i in range(4)]
    assert identity['supervisor']['affinity']=='0'
    births={(identity['supervisor']['pid'],identity['supervisor']['start'])};prior=identity['started'];summary=[]
    for run in identity['runs']:
        assert run['code']==0 and prior<=run['started']<run['ended']<=identity['ended'];prior=run['ended']
        assert 0<run['seconds']<600 and run['preflight_available']>=4*1024**3 and run['preflight_disk']>=128*1024**2
        assert run['flags']=={} and run['members'][str(run['pid'])]==run['start']
        rows=samples[run['name']];assert rows and len(rows)==run['samples'];seconds=0;peak=0;cpu=None
        for row in rows:
            assert seconds<=row['seconds']<run['seconds'] and row['seconds']-seconds<10;seconds=row['seconds']
            assert row['available_memory']>=LIMITS['available_memory'];rss=sum(m['rss'] for m in row['members']);assert 0<=rss<LIMITS['rss'];peak=max(peak,rss)
            assert len({m['pid'] for m in row['members']})==len(row['members'])
            for m in row['members']:
                assert m['affinity']=='2' and m['group']==run['pid'] and m['start']>=run['start'] and m['rss']>=0 and m['cpu_seconds']>=0
                assert run['members'][str(m['pid'])]==m['start'];births.add((m['pid'],m['start']))
            assert set(row['cpu'])=={'cpu','cpu2'}
            for name,values in row['cpu'].items():
                assert len(values)==10 and all(v>=0 for v in values)
                if cpu:assert all(v>=old for v,old in zip(values,cpu[name]))
            cpu=row['cpu']
        assert run['seconds']-seconds<10 and peak==run['peak_rss']
        births.add((run['pid'],run['start']));births.update((int(p),s) for p,s in run['members'].items())
        summary.append(dict(name=run['name'],seconds=run['seconds'],samples=len(rows),peak_rss=peak,foreign_activity=run['foreign_activity'],
                            cpu_delta={k:[b-a for a,b in zip(rows[0]['cpu'][k],rows[-1]['cpu'][k])] for k in ['cpu','cpu2']}))
    return dict(runs=summary,births=[dict(pid=p,start=s) for p,s in sorted(births)])
