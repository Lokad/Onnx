import unittest
from checks import metadata_delta

METHOD='MEMBER Lokad.Onnx.ComputationalGraph Method Int32 PrepareOwnedMatMulWeights()'
FRIEND='[System.Runtime.CompilerServices.InternalsVisibleToAttribute("Lokad.Onnx.Data")]'


class MetadataTests(unittest.TestCase):
    def fixture(self):
        return dict(assembly='Lokad.Onnx.dll',public_surface=['existing'],
            public_surface_after=['existing',METHOD,METHOD+' FLAGS Public, HideBySig'],
            assembly_attributes_before=['retained',FRIEND],assembly_attributes_after=['retained'],public_surface_equal=False)

    def test_exact_promotion_and_friend_removal_pass(self):
        self.assertTrue(metadata_delta(self.fixture()))

    def test_other_public_changes_fail(self):
        for key,value in [('public_surface_after',['existing',METHOD]),
                          ('public_surface_after',['other',METHOD,METHOD+' FLAGS Public, HideBySig']),
                          ('public_surface_equal',True)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                row=self.fixture();row[key]=value;metadata_delta(row)

    def test_only_one_friend_attribute_can_be_removed(self):
        for attributes in [['retained',FRIEND],[],['retained','new']]:
            with self.subTest(attributes=attributes),self.assertRaises(AssertionError):
                row=self.fixture();row['assembly_attributes_after']=attributes;metadata_delta(row)

    def test_data_metadata_must_remain_identical(self):
        row=dict(assembly='Lokad.Onnx.Data.dll',public_surface=['existing'],public_surface_after=['existing'],
            assembly_attributes_before=['retained'],assembly_attributes_after=['retained'],public_surface_equal=True)
        self.assertTrue(metadata_delta(row))
        row['public_surface_after'].append(METHOD)
        with self.assertRaises(AssertionError):metadata_delta(row)


if __name__=='__main__':unittest.main()
