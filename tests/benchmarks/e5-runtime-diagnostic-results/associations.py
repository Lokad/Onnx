"""Match complete runtime event pairs; retain unmatched boundary records."""
import bisect,collections

def associations(calls,events):
    starts=[c['begin_ms'] for c in calls];ends=[c['end_ms'] for c in calls]
    def overlap(start,end):
        i=max(0,bisect.bisect_left(ends,start));found=[]
        while i<len(calls) and starts[i]<end:
            delta=max(0,min(end,ends[i])-max(start,starts[i]))
            if delta:found.append((i,delta))
            i+=1
        return found
    def location(at):
        i=bisect.bisect_right(starts,at)-1
        return i if i>=0 and at<=ends[i] else -1
    queues=collections.defaultdict(list);loads=[];compiles=[];pending_pauses={};pauses=[];unmatched=[]
    for e in events:
        if e['provider']!='Microsoft-Windows-DotNETRuntime':continue
        p=e['payload']
        if e['name']=='Method/JittingStarted':queues[(e['thread'],p['MethodID'])].append(e)
        elif e['name']=='Method/LoadVerbose':
            key=(e['thread'],p['MethodID'])
            if queues[key]:
                start=queues[key].pop(0);touched=overlap(start['ms'],e['ms'])
                compiles.append(dict(method=p['MethodName'],namespace=p['MethodNamespace'],thread=e['thread'],tier=p['OptimizationTier'],
                    start_ms=start['ms'],end_ms=e['ms'],overlap_ms=sum(d for i,d in touched),calls=len(touched)))
            else:unmatched.append(dict(kind='load_without_start',event=e))
            if p['MethodNamespace'].startswith('Lokad.Onnx.'):
                loads.append(dict(ms=e['ms'],call=location(e['ms']),thread=e['thread'],method=p['MethodName'],namespace=p['MethodNamespace'],
                    tier=p['OptimizationTier'],bytes=int(p['MethodSize']),address=p['MethodStartAddress']))
        elif e['name']=='GC/SuspendEEStart':
            assert e['thread'] not in pending_pauses;pending_pauses[e['thread']]=e
        elif e['name']=='GC/RestartEEStop':
            start=pending_pauses.pop(e['thread'],None)
            if start is None:unmatched.append(dict(kind='restart_without_suspend',event=e));continue
            touched=overlap(start['ms'],e['ms'])
            pauses.append(dict(thread=e['thread'],start_ms=start['ms'],end_ms=e['ms'],reason=start['payload']['Reason'],
                overlap_ms=sum(d for i,d in touched),calls=len(touched)))
    unmatched += [dict(kind='start_without_load',event=e) for items in queues.values() for e in items]
    unmatched += [dict(kind='suspend_without_restart',event=e) for e in pending_pauses.values()]
    return dict(loads=loads,compilations=compiles,suspensions=pauses,unmatched=unmatched)
