"""Permit only the two source-verified cold-to-executed tensor name changes."""
import copy


def validate_transition(before,after):
    expected=copy.deepcopy(before)
    for graph,key in [('first','folded:Transpose_1010'),('past','folded:Transpose_801')]:
        output='model.decoder.embed_tokens.weight_transposed'
        entries=[r for r in expected[graph]['initializers'] if r['name']==key]
        assert len(entries)==1 and entries[0]['tensor_name']==key
        nodes=[n for n in expected[graph]['nodes'] if n['Name']==key.removeprefix('folded:')]
        assert len(nodes)==1 and nodes[0]['op']=='Transpose' and nodes[0]['Outputs']==[output]
        entries[0]['tensor_name']=output
    assert expected==after,'Unexpected decoder snapshot change'
