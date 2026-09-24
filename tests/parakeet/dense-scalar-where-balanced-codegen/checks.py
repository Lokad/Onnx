"""Check the complete timer boundary and balanced journal independently of clocks."""


def inspect_il(value):
    assert value['passed'] and value['implementation_flags']==8
    assert value['method']=='Int64 MeasureBatch(Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.OpResult[])'
    assert value['locals']==['System.Int64','System.Int32'] and value['exceptions']==0
    rows=value['instructions'];codes=[r['opcode'] for r in rows]
    prefix=['call','stloc.0','ldc.i4.0','stloc.1','br.s']
    # Roslyn may materialize the array element address or emit a value-type store.
    forms=[['ldarg.3','ldloc.1','ldelema','ldarg.0','ldarg.1','ldarg.2','ldnull','call','stobj'],
           ['ldarg.3','ldloc.1','ldarg.0','ldarg.1','ldarg.2','ldnull','call','stelem']]
    suffix=['ldloc.1','ldc.i4.1','add','stloc.1','ldloc.1','ldarg.3','ldlen','conv.i4','blt.s','call','ldloc.0','sub','ret']
    assert any(codes==prefix+form+suffix for form in forms),codes
    calls=[r['operand'] for r in rows if r['opcode']=='call']
    assert calls==['System.Diagnostics.Stopwatch:Int64 GetTimestamp()',
        'Lokad.Onnx.CPUExecutionProvider:Lokad.Onnx.OpResult Where(Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ExecutionOptions)',
        'System.Diagnostics.Stopwatch:Int64 GetTimestamp()'],calls
    for r in rows:
        if r['opcode'] in ['ldelema','stobj','stelem']: assert r['operand']=='Lokad.Onnx.OpResult'
    assert rows[4]['operand']==rows[-9]['offset']
    assert rows[-5]['opcode']=='blt.s' and rows[-5]['operand']==rows[5]['offset']
    assert len({r['offset'] for r in rows})==len(rows) and all(a['offset']<b['offset'] for a,b in zip(rows,rows[1:]))
    assert rows[-1]['offset']==value['il_bytes']-1
    return dict(passed=True,exact_timed_loop=True,no_inlining_only=True,product_calls=1,timestamps=2,
                il_bytes=value['il_bytes'],instructions=len(rows),exceptions=0)


def expected_journal(rows):
    for iteration in range(600):
        for row in rows: yield row['clocks'][iteration]
    for row in rows:
        yield from row['clocks'][600:780]


def check_journal(clocks,rows):
    assert len(rows)==220 and len(clocks)==171600
    assert all(a==b for a,b in zip(clocks,expected_journal(rows),strict=True))
