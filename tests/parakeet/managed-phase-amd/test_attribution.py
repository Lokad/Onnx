"""Refuse missing/overlapping work before interpreting managed node clocks."""
import json
from pathlib import Path
import tempfile
import unittest
from audit import attribute, PHASES


def fixture(folder):
    graphs={g:dict(nodes=[dict(id=1,name='node',op='MatMul',inputs=[],outputs=[],constant_inputs=[])]) for g in PHASES}
    (folder/'graphs.json').write_text(json.dumps(graphs))
    rows=[]
    for index in range(80):
        start=index*1000;iteration=index//20
        calls=[]
        for i,g in enumerate(['nemo128.onnx','encoder-model.onnx']+['decoder_joint-model.onnx']*60):
            a=start+10+i*10;b=a+5
            calls.append(dict(graph=g,start_ticks=a,end_ticks=b,nodes=[dict(NodeId=1,Op=0,StartTicks=a+1,EndTicks=b-1)]))
        name=str(index%20)
        rows.append(dict(name=name,pass_=iteration,phase='warmup' if iteration==0 else 'measured',frequency=1000,
                         start_ticks=start,end_ticks=start+900,result=dict(decoder_calls=60)))
        rows[-1]['pass']=rows[-1].pop('pass_')
        (folder/f'phase-{index:03}.json').write_text(json.dumps(dict(name=name,pass_=iteration,frequency=1000,mode='wall',calls=calls)).replace('"pass_"','"pass"'))
    return dict(records=rows)


class Attribution(unittest.TestCase):
    def test_warmup_excluded_and_remainder_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);result=attribute(fixture(folder),folder,'wall')
            self.assertEqual(result['call_counts'],dict(frontend=60,encoder=60,decoder=3600))
            self.assertEqual(sum(n['calls'] for n in result['node_rows']),3720)
            self.assertAlmostEqual(sum(result['phase_seconds'].values())+result['remainder_seconds'],result['corpus_seconds'])

    def test_missing_node_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);result=fixture(folder);path=folder/'phase-020.json';value=json.loads(path.read_text())
            value['calls'][1]['nodes']=[];path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):attribute(result,folder,'wall')

    def test_node_outside_phase_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);result=fixture(folder);path=folder/'phase-020.json';value=json.loads(path.read_text())
            call=value['calls'][1];call['nodes'][0]['EndTicks']=call['end_ticks']+1;path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):attribute(result,folder,'wall')


if __name__=='__main__':unittest.main()
