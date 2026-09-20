import io
import hashlib
from pathlib import Path
import tarfile
import tempfile
import unittest
from collect import extract


class CollectionTests(unittest.TestCase):
    def test_exact_inventory_and_unsafe_archives(self):
        content=b'complete result\n'
        expected={'worker/result.json':dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i,kind in enumerate(['valid','duplicate','absolute','parent','backslash','link','missing','changed']):
                archive=root/(kind+'.tar.gz')
                with tarfile.open(archive,'w:gz') as stream:
                    if kind!='missing':
                        name={'absolute':'/worker/result.json','parent':'../worker/result.json','backslash':'worker\\result.json'}.get(kind,'worker/result.json')
                        member=tarfile.TarInfo(name);member.size=len(content)
                        if kind=='link':member.type=tarfile.SYMTYPE;member.linkname='../elsewhere';member.size=0;stream.addfile(member)
                        else:
                            stream.addfile(member,io.BytesIO(content if kind!='changed' else b'changed result!\n'))
                            if kind=='duplicate':stream.addfile(member,io.BytesIO(content))
                if kind=='valid':
                    extract(archive,root/'valid',expected)
                    self.assertEqual((root/'valid/worker/result.json').read_bytes(),content)
                else:
                    with self.assertRaises(AssertionError):extract(archive,root/('out'+str(i)),expected)


if __name__=='__main__':unittest.main()
