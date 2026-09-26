"""Reject missing warmups, altered clocks, flags and observer bodies."""
import base64
from collections import Counter
import copy
import json
from pathlib import Path
import struct
import unittest
from audit import events_checked
from compatibility import QUALIFIED, CANDIDATE, OBSERVER, read, reconcile
from scope import consumer, supervisor, attribute_module


def fixture():
    events = []; anchors = []; records = []
    def event(provider, name, ms, raw, payload):
        events.append(dict(index=len(events),provider=provider,name=name,id=1,pid=99,thread=99,ms=ms,
            rawLength=len(raw),rawBase64=base64.b64encode(raw).decode(),payload=payload))
    def clock(counter):
        anchors.append(dict(before=counter,after=counter+100,thread=99))
        event('Lokad-Parakeet-Clock','Anchor',(counter+50)/1e6+10,
              struct.pack('<q',counter),dict(counter=str(counter)))
    event('Microsoft-Windows-DotNETRuntime','Method/LoadVerbose',0,b'',{})
    clock(1000000)
    for i in range(80):
        start=1000000000*(i+1);end=start+1000000;name='clip-'+str(i%20);iteration=i//20
        records.append(dict(name=name,pass_index=iteration,frequency=10**9,start_ticks=start,end_ticks=end,thread_id=99))
        records[-1]['pass']=records[-1].pop('pass_index')
        clock(start-1000)
        for phase,tick in [('begin',start-500),('end',end+500)]:
            event('Lokad-Pyannote-Diagnostic','Boundary',tick/1e6+10,
                  (phase+'\0'+name+'\0').encode('utf-16-le')+struct.pack('<i',iteration),
                  {'phase':phase,'name':name,'pass':str(iteration)})
        clock(end+1000)
    summary=dict(complete=True,lost=0,clr_events=1,protocol='all-event-records-v1',recorded=len(events),
                 allCounts=dict(Counter(e['provider']+':'+e['name'] for e in events)))
    return events,summary,dict(frequency=10**9,anchors=anchors),records,99


class DiagnosticTests(unittest.TestCase):
    def test_all_warmup_and_measured_markers(self):
        result=events_checked(*fixture())
        self.assertEqual((result['markers'],result['anchors']),(160,161))
        self.assertLess(result['uncertainty_ms'],.005)

    def test_lost_events_fail(self):
        args=fixture();args[1]['lost']=1
        with self.assertRaises(AssertionError):events_checked(*args)

    def test_wrong_warmup_marker_fails(self):
        args=fixture();next(e for e in args[0] if e['provider']=='Lokad-Pyannote-Diagnostic')['payload']['pass']='1'
        with self.assertRaises(AssertionError):events_checked(*args)

    def test_inconsistent_clock_anchor_fails(self):
        args=fixture();next(e for e in args[0] if e['provider']=='Lokad-Parakeet-Clock')['ms']+=1
        with self.assertRaises(AssertionError):events_checked(*args)

    def test_implementation_flag_change_fails(self):
        root=read(QUALIFIED/'collected/inventory/instructions.json')
        candidate=read(CANDIDATE/'collected/inventory/instructions.json')
        observer=read(OBSERVER/'build-collected/inventory/instructions.json')
        self.assertTrue(reconcile(root,candidate,observer)['selected_runtime_flags_pending'])
        row=candidate['observations'][0];key=next(k for k in row['method_flags_after'] if '::PadCore::' in k)
        row['method_flags_after'][key]^=512
        with self.assertRaises(AssertionError):reconcile(root,candidate,observer)

    def test_changed_observer_original_body_fails(self):
        root=read(QUALIFIED/'collected/inventory/instructions.json')
        candidate=read(CANDIDATE/'collected/inventory/instructions.json')
        observer=read(OBSERVER/'build-collected/inventory/instructions.json')
        row=observer['observations'][-1];row['normalized_methods'][next(iter(row['normalized_methods']))]='changed'
        with self.assertRaises(AssertionError):reconcile(root,candidate,observer)


if __name__=='__main__':unittest.main()
