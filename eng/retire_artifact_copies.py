"""Bounded parallel file retirement using the reviewed Windows artifact manifest.

Resumes the native PowerShell journal; no recursive deletion and no shell-built
file operations. Four readers avoid cmdlet overhead across 210,000 tiny files.
"""
import concurrent.futures,hashlib,json,os,stat,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 manifest_path=Path(sys.argv[1]).resolve();base=manifest_path.parent
 artifact_root=os.path.normcase(os.path.abspath(ROOT/'artifacts'));prefix=artifact_root+os.sep
 assert os.path.normcase(str(base)).startswith(prefix)
 manifest=json.loads(manifest_path.read_text());assert Path(manifest['repo'])==ROOT
 fingerprint=digest(manifest_path);keep=set(manifest['keep']);extensions=set(manifest['extensions'])
 tracked=set(subprocess.run(['git','-C',str(ROOT),'ls-files','-z','--','artifacts'],check=True,capture_output=True).stdout.decode().split('\0'))
 seen=set()
 for entry in manifest['files']:
  path=entry['path'];parts=path.split('/');target=os.path.normcase(os.path.abspath(ROOT/path))
  assert target.startswith(prefix) and target not in seen and path not in tracked
  assert len(parts)>=3 and parts[0]=='artifacts' and '20260923' not in parts[1] and parts[1] not in keep
  assert Path(path).suffix.lower() in extensions or '.so.' in parts[-1]
  seen.add(target)
 assert not (base/'completed.json').exists()
 prior={}
 for name in ['retired.jsonl','retired-python.jsonl']:
  p=base/name
  if p.exists():
   for line in p.read_text().splitlines():
    entry=json.loads(line);assert entry['manifest_sha256']==fingerprint
    if entry['path'] in prior:assert entry==prior[entry['path']]
    prior[entry['path']]=entry
 checked=set();lock=threading.Lock();stop=threading.Event();started=time.time();processed=0;retired_bytes=0
 script_hash=digest(__file__)
 with (base/'retired-python.jsonl').open('a',encoding='utf8',newline='\n') as journal:
  def retire(entry):
   if stop.is_set():return None
   try:
    relative=entry['path'];target=os.path.normcase(os.path.abspath(ROOT/relative));old=prior.get(relative)
    if old and not os.path.exists(target):
     assert old['bytes']==entry['bytes'];return entry['bytes']
    metadata=os.stat(target,follow_symlinks=False)
    assert stat.S_ISREG(metadata.st_mode) and not metadata.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    parent=os.path.dirname(target)
    while parent.startswith(os.path.normcase(str(ROOT))):
     if parent in checked:break
     attributes=os.stat(parent,follow_symlinks=False)
     assert stat.S_ISDIR(attributes.st_mode) and not attributes.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,parent
     checked.add(parent);parent=os.path.dirname(parent)
    assert metadata.st_size==entry['bytes'] and metadata.st_mtime_ns==entry['mtime_ns'],relative
    sha=digest(target)
    if old:assert old['sha256']==sha,relative
    else:
     receipt=dict(path=relative,bytes=entry['bytes'],mtime_ns=entry['mtime_ns'],sha256=sha,manifest_sha256=fingerprint)
     with lock:journal.write(json.dumps(receipt,separators=(',',':'))+'\n');journal.flush()
    # Recheck after hashing and before unlinking this exact regular file.
    again=os.stat(target,follow_symlinks=False)
    assert (again.st_ino,again.st_size,again.st_mtime_ns)==(metadata.st_ino,metadata.st_size,metadata.st_mtime_ns)
    os.unlink(target);assert not os.path.exists(target)
    return entry['bytes']
   except BaseException:stop.set();raise
  success=False
  try:
   last=time.monotonic()
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for size in pool.map(retire,manifest['files']):
     assert size is not None;processed+=1;retired_bytes+=size
     if time.monotonic()-last>=15:
      print(f'Retired {processed:,}/{len(manifest["files"]):,} files, {retired_bytes/1e9:.2f} GB',flush=True);last=time.monotonic()
   assert processed==len(manifest['files']) and retired_bytes==manifest['planned_bytes'];success=True
  finally:
   receipt=dict(passed=success,manifest_sha256=fingerprint,script_sha256=script_hash,processed=processed,bytes=retired_bytes,started=started,ended=time.time(),pid=os.getpid())
   (base/'latest-python.json').write_text(json.dumps(receipt,indent=2)+'\n')
   if success:(base/'completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
