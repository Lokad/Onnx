"""Preserve M41 arithmetic; change only four float implementation flags."""
from base_checks import inventory as original_inventory
METHODS=['PackPanelsB','mm_unsafe_vectorized_intrinsics_2x4packed_bump','mm_unsafe_vectorized_intrinsics_3x4packed','mm_unsafe_vectorized_intrinsics']
def kernels(flags):
 result=[]
 for name in METHODS:
  found=[key for key in flags if key.startswith('Lokad.Onnx.MathOps::'+name+'::') and 'Single*' in key and 'Double*' not in key]
  assert len(found)==1,(name,found);result.extend(found)
 return result
def flags(row,parent=False):
 before=dict(row['method_flags_before']);after=dict(row['method_flags_after']);rename=row['compiler_rename']
 if rename:
  assert not parent and after[rename['newKey']]==before[rename['oldKey']]
  after[rename['oldKey']]=after.pop(rename['newKey'])
 assert set(before)<=set(after)
 expected=kernels(before) if row['assembly']=='Lokad.Onnx.dll' else []
 for key in expected:assert before[key]==0 and after[key]==512,key
 if not parent and row['assembly']=='Lokad.Onnx.dll':
  wrapper,=row['differences'];expected.append(wrapper);assert before[wrapper]==0 and after[wrapper]==256
  assert set(after)-set(before)==set(row['added']) and len(row['added'])==2
  assert all(after[key]==8 for key in row['added'])
 else:assert set(after)==set(before)
 assert {key for key in before if before[key]!=after[key]}==set(expected)
 return expected
def inventory(value,measured,built):
 result=original_inventory(value,measured,built)
 result['implementation_flags']={row['assembly']:flags(row) for row in value['observations']}
 return result
def previous_inventory(value,previous,built):
 assert value['inventory_complete'] and len(value['observations'])==2
 rows=[]
 for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3181),('Lokad.Onnx.Data.dll',697)],strict=True):
  assert row['assembly']==name and row['methods']==row['unchanged_methods']==len(row['normalized_methods'])==count
  assert row['before_sha256']==previous[name]['sha256'] and row['after_sha256']==built[name]['sha256']
  assert row['public_surface_equal'] and row['compiler_rename'] is None
  assert not row['removed'] and not row['added'] and not row['differences'] and not row['candidate_methods']
  rows.append(dict(assembly=name,identical_method_bodies=count,flag_changes=flags(row,True)))
 return dict(passed=True,observations=rows,only_four_float_flags=True)
