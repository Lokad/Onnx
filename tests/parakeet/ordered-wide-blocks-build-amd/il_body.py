"""Retained decoding of branch widths and instruction/exception boundaries."""
import json


def normalized_body(value):
    """Normalize encoded widths while retaining every branch and exception target."""
    body=json.loads(value);ins=body['instructions'];assert ins and ins[-1]['opcode']=='ret'
    positions={i['offset']:n for n,i in enumerate(ins)};positions[ins[-1]['offset']+1]=len(ins)
    result=[]
    for index,item in enumerate(ins):
        op=item['opcode'];operand=item['operand']
        if op in ['ldc.i4','ldc.i4.s']:
            operand=int.from_bytes(bytes.fromhex(operand),'little',signed=True);op='ldc.i4'
        elif op!='break' and op.startswith(('br','beq','bne','bge','bgt','ble','blt','leave')):
            target=ins[index+1]['offset']+int.from_bytes(bytes.fromhex(operand),'little',signed=True)
            operand=positions[target];op=op.removesuffix('.s')
        elif op=='switch':
            raw=bytes.fromhex(operand);assert len(raw)%4==0
            operand=[positions[ins[index+1]['offset']+int.from_bytes(raw[n:n+4],'little',signed=True)] for n in range(0,len(raw),4)]
        result.append(dict(opcode=op,operand=operand))
    regions=[]
    for original in body['exceptions']:
        region=dict(original)
        for prefix in ['Try','Handler']:
            start=original[prefix+'Offset'];end=start+original[prefix+'Length']
            region[prefix+'Offset']=positions[start];region[prefix+'Length']=positions[end]-positions[start]
        if region['filter']!=-1:region['filter']=positions[region['filter']]
        regions.append(region)
    return dict(InitLocals=body['InitLocals'],MaxStackSize=body['MaxStackSize'],locals=body['locals'],exceptions=regions,instructions=result)

