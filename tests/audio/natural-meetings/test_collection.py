"""Collection refuses dangerous archives before creating the destination."""
from pathlib import Path
import hashlib
import io
import tarfile
import tempfile
import unittest
from collect import extract


class CollectionTests(unittest.TestCase):
    def archive(self,path,names,symlink=False):
        with tarfile.open(path,'w:gz') as stream:
            for name in names:
                item=tarfile.TarInfo(name);item.size=3
                if symlink:item.type=tarfile.SYMTYPE;item.linkname='../outside'
                stream.addfile(item,io.BytesIO(b'abc'))

    def test_exact_inventory_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'a.tar.gz';self.archive(archive,['nested/result.json'])
            expected={'nested/result.json':dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())}
            extract(archive,root/'result',expected)
            self.assertEqual((root/'result/nested/result.json').read_bytes(),b'abc')
            with self.assertRaises(AssertionError):extract(archive,root/'result',expected)

    def test_refuse_traversal_links_duplicates_and_missing_members(self):
        for names,symlink,expected_names in [(['../outside'],False,['../outside']),
            (['/absolute'],False,['/absolute']),(['C:/absolute'],False,['C:/absolute']),
            (['a\\b'],False,['a\\b']),(['a'],True,['a']),(['a','a'],False,['a']),
            (['a'],False,['a','missing'])]:
            with self.subTest(names=names,symlink=symlink),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);archive=root/'a.tar.gz';self.archive(archive,names,symlink)
                expected={n:dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest()) for n in expected_names}
                with self.assertRaises(AssertionError):extract(archive,root/'result',expected)
                self.assertFalse((root/'result').exists())


if __name__=='__main__':unittest.main()
