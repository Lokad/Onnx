"""Derive arithmetic scope from pinned ONNX weights and convolution attributes."""
import onnx
from protocol import pin, read


def eligible(shape,attributes,data_type):
    return (data_type==onnx.TensorProto.FLOAT and len(shape)==4 and shape[0]>=32 and shape[0]%16==0
        and shape[1]>=16 and shape[1]%16==0 and shape[2:]==[3,3] and attributes.get('group',1)==1
        and attributes.get('strides',[1,1])==[1,1] and attributes.get('dilations',[1,1])==[1,1]
        and attributes.get('pads',[0,0,0,0])==[1,1,1,1])


def census(root,manifest):
    rows=[]
    for item in manifest['models']:
        asset=item['assets'][0];path=root/asset['file']
        assert pin(path)=={k:asset[k] for k in ['bytes','sha256']}
        model=onnx.load(str(path),load_external_data=False);weights={w.name:w for w in model.graph.initializer}
        inputs={n.name for n in model.graph.input};outputs={n.name for n in model.graph.output}
        uses={}
        for node in model.graph.node:
            for i,name in enumerate(node.input):uses.setdefault(name,[]).append((node.op_type,i))
        nodes=[];convolutions=0
        for node in model.graph.node:
            if node.op_type!='Conv':continue
            convolutions+=1;weight=weights.get(node.input[1]);attributes={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            if weight is None or weight.name in inputs or weight.name in outputs:continue
            if not all(op=='Conv' and i==1 for op,i in uses[weight.name]):continue
            shape=list(weight.dims)
            if eligible(shape,attributes,weight.data_type):
                nodes.append(dict(node=node.name,weight=weight.name,shape=shape,group=attributes.get('group',1),
                    strides=attributes.get('strides',[1,1]),dilations=attributes.get('dilations',[1,1]),pads=attributes.get('pads',[0,0,0,0])))
        rows.append(dict(model=item['key'],asset=asset,convolutions=convolutions,eligible=nodes))
    return dict(models=rows,models_with_eligible_convolutions=[r['model'] for r in rows if r['eligible']],
        policy='Every native array retains the original 1e-4 bound. Cross-product exactness remains mandatory for models without eligible changed-arithmetic convolutions. All differences are reported; no model-specific dispatch.')
