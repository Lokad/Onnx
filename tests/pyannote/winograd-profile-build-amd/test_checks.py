"""Use the actual qualified consumer inventory to test exact hash-only adaptation."""
import copy
from pathlib import Path
import unittest
from checks import inventory,OLD_CORE,OLD_DATA
from protocol import read

ROOT=Path(__file__).resolve().parents[3]
OLD=read(ROOT/'artifacts/pyannote-prepared-profile-amd-v2-20260922/instructions.json')
CORE='208371f620fcfce5ff709db54525fa158b4c9442685430afbeb730619e1ded81'
DATA='b935837024c3ffd67e322ba1fe44405b24b7da221779cf00856f090df08693f5'
PRODUCT={'Lokad.Onnx.dll':{'sha256':CORE},'Lokad.Onnx.Data.dll':{'sha256':DATA}}


def example():
    value=copy.deepcopy(OLD);row=value['observations'][0];key,=row['differences']
    row['normalized_methods'].update(row['candidate_methods'])
    row['candidate_methods']={key:row['normalized_methods'][key].replace(OLD_CORE,CORE).replace(OLD_DATA,DATA)}
    row['before_sha256']='previous';row['after_sha256']='built'
    return value


def check(value):return inventory(value,{'sha256':'previous'},{'sha256':'built'},PRODUCT)


class ScopeTests(unittest.TestCase):
    def test_actual_compiled_census(self):self.assertTrue(check(example())['passed'])
    def test_other_changes_rejected(self):
        for change in ['extra','public','guard','core','data','missing','identity']:
            with self.subTest(change=change):
                value=example();row=value['observations'][0];key,=row['differences']
                if change=='extra':row['added']=['new method']
                elif change=='public':row['public_surface_equal']=False
                elif change=='guard':row['candidate_methods'][key]+=' '
                elif change=='core':row['candidate_methods'][key]=row['candidate_methods'][key].replace(CORE,OLD_CORE)
                elif change=='data':row['candidate_methods'][key]=row['candidate_methods'][key].replace(DATA,OLD_DATA)
                elif change=='missing':row['normalized_methods'].pop(key)
                else:row['after_sha256']='wrong'
                with self.assertRaises(AssertionError):check(value)


if __name__=='__main__':unittest.main(verbosity=2)
