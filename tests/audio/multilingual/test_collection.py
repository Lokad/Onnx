from pathlib import Path
import hashlib
import io
import tarfile
import tempfile
import unittest
from collect_amd import extract


class CollectionTests(unittest.TestCase):
    def test_exact_bytes_and_inventory_then_unsafe_or_incomplete_archives_refused(self):
        content=b'complete retained evidence\n';expected={'run/case.json':dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def archive(name,entries):
                path=root/name
                with tarfile.open(path,'w:gz') as stream:
                    for entry,data,kind in entries:
                        member=tarfile.TarInfo(entry);member.type=kind;member.size=len(data) if kind==tarfile.REGTYPE else 0
                        if kind==tarfile.SYMTYPE:member.linkname='../outside'
                        stream.addfile(member,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
                return path
            good=archive('good.tgz',[('run/case.json',content,tarfile.REGTYPE)])
            extract(good,root/'good',expected);self.assertEqual((root/'good/run/case.json').read_bytes(),content)
            with self.assertRaises(AssertionError):extract(good,root/'good',expected)
            cases=[[],[('extra',content,tarfile.REGTYPE)],
                [('run/case.json',content,tarfile.REGTYPE)]*2,
                [('../escape',content,tarfile.REGTYPE)],[('/absolute',content,tarfile.REGTYPE)],
                [('C:/absolute',content,tarfile.REGTYPE)],[('run\\case.json',content,tarfile.REGTYPE)],
                [('run/case.json',content,tarfile.SYMTYPE)],
                [('run/case.json',content[:-1],tarfile.REGTYPE)],
                [('run/case.json',b'X'+content[1:],tarfile.REGTYPE)]]
            for number,entries in enumerate(cases):
                with self.assertRaises(AssertionError):extract(archive(str(number)+'.tgz',entries),root/str(number),expected)


if __name__=='__main__':unittest.main()
