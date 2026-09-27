"""Read-only Windows allocation inventory, with both per-path and unique-file totals."""
import argparse
from collections import defaultdict
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import stat
import time

ROOT=Path(__file__).resolve().parents[1]


class StandardInfo(ctypes.Structure):
    _fields_=[('allocated',ctypes.c_longlong),('logical',ctypes.c_longlong),
              ('links',wintypes.DWORD),('delete_pending',ctypes.c_ubyte),('directory',ctypes.c_ubyte)]


def main():
    assert os.name=='nt', 'This inventory uses Windows FILE_STANDARD_INFO.'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();destination=args.output.resolve()
    assert destination.is_relative_to(ROOT) and not destination.exists() and destination.parent.is_dir()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    query=kernel.GetFileInformationByHandleEx
    query.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];query.restype=wintypes.BOOL
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
    disk=kernel.GetDiskFreeSpaceW
    disk.argtypes=[wintypes.LPCWSTR,*([ctypes.POINTER(wintypes.DWORD)]*4)];disk.restype=wintypes.BOOL
    sectors,bytes_per_sector,free,total=[wintypes.DWORD() for _ in range(4)]
    assert disk(ROOT.drive+'\\',ctypes.byref(sectors),ctypes.byref(bytes_per_sector),ctypes.byref(free),ctypes.byref(total))
    cluster=sectors.value*bytes_per_sector.value;assert cluster>0
    root='\\\\?\\'+str(ROOT);pending=[root];groups=defaultdict(lambda:dict(logical=0,allocated=0,files=0,compressed_files=0))
    aliases=[];fallbacks=[];hardlink_paths=0
    linked_files=set();unique_allocated=0;unidentified_links=0
    while pending:
        folder=pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                info=entry.stat(follow_symlinks=False);relative=Path(os.path.relpath(entry.path,root))
                if stat.S_ISLNK(info.st_mode) or info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    aliases.append(relative.as_posix());continue
                if stat.S_ISDIR(info.st_mode):pending.append(entry.path);continue
                assert stat.S_ISREG(info.st_mode),relative
                handle=create(entry.path,0x80,7,None,3,0,None)
                actual=StandardInfo();error=None
                if handle==ctypes.c_void_p(-1).value:error=ctypes.get_last_error()
                else:
                    try:
                        if not query(handle,1,ctypes.byref(actual),ctypes.sizeof(actual)):error=ctypes.get_last_error()
                    finally:assert close(handle)
                if error is None:
                    assert not actual.directory and actual.logical==info.st_size and actual.allocated>=0,relative
                    allocated=actual.allocated;hardlink_paths+=actual.links>1
                else:
                    # Locked files retain a conservative cluster-rounded estimate.
                    assert error in [5,32,33],(relative,error)
                    allocated=((info.st_size+cluster-1)//cluster)*cluster
                    fallbacks.append(dict(path=relative.as_posix(),winerror=error,logical=info.st_size,allocated_upper_bound=allocated))
                links=actual.links if error is None else info.st_nlink
                if links<=1:
                    unique_allocated+=allocated
                elif info.st_ino and info.st_dev:
                    identity=(info.st_dev,info.st_ino)
                    if identity not in linked_files:
                        linked_files.add(identity);unique_allocated+=allocated
                else:
                    # Never merge unidentified files: retain a conservative bound.
                    unique_allocated+=allocated;unidentified_links+=1
                group=groups[relative.parts[0]];group['logical']+=info.st_size;group['allocated']+=allocated;group['files']+=1
                group['compressed_files']+=bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_COMPRESSED)
    value=dict(checked=time.time(),logical_bytes=sum(g['logical'] for g in groups.values()),
        allocated_bytes=sum(g['allocated'] for g in groups.values()),files=sum(g['files'] for g in groups.values()),
        compressed_files=sum(g['compressed_files'] for g in groups.values()),groups=dict(groups),
        hardlinked_paths_counted_separately=hardlink_paths,aliases_excluded=aliases,cluster_bytes=cluster,
        unique_allocated_bytes=unique_allocated,identified_hardlinked_files=len(linked_files),
        unidentified_hardlink_paths_counted_separately=unidentified_links,
        locked_file_upper_bounds=fallbacks,
        accounting='allocated_bytes and groups retain per-path FILE_STANDARD_INFO AllocationSize totals, counting hardlinked paths separately. unique_allocated_bytes counts identified hardlinked files once using volume/file identifiers; unidentified hardlinks remain conservatively separate. Directory aliases are excluded and locked files conservatively rounded to clusters. Includes file data allocation; filesystem metadata and unrelated volume data are outside repository files. No compression or filesystem mutation occurs except writing this new receipt after inventory.')
    value['within_50_decimal_gb_allocated']=value['allocated_bytes']<50_000_000_000
    assert value['unique_allocated_bytes']<=value['allocated_bytes']
    value['within_50_decimal_gb_unique_allocated']=value['unique_allocated_bytes']<50_000_000_000
    with destination.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['groups','locked_file_upper_bounds','accounting']},indent=2))


if __name__=='__main__':main()
