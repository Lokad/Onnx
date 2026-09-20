"""Exercise the actual Windows sharing violation that stopped the first run."""
from pathlib import Path
import ctypes
import json
import os
import tempfile
import threading
import time
import unittest
from supervise import save_status


@unittest.skipUnless(os.name=='nt','Windows file-sharing contract')
class StatusTests(unittest.TestCase):
    def test_transient_reader_is_retried_and_persistent_lock_still_fails(self):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
        kernel.CreateFileW.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'identity.json';save_status(path,dict(version=1))
            def lock():
                handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0,None)
                self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
                return handle
            handle=lock()
            def release():
                time.sleep(.08);kernel.CloseHandle(handle)
            thread=threading.Thread(target=release);thread.start()
            start=time.monotonic();save_status(path,dict(version=2));elapsed=time.monotonic()-start;thread.join()
            self.assertGreaterEqual(elapsed,.05)
            self.assertEqual(json.loads(path.read_text()),dict(version=2))
            handle=lock()
            try:
                with self.assertRaises(PermissionError):save_status(path,dict(version=3),timeout=.03)
                self.assertEqual(json.loads(path.read_text()),dict(version=2))
            finally:kernel.CloseHandle(handle)
            save_status(path,dict(version=3));self.assertFalse(path.with_suffix('.tmp').exists())
            self.assertEqual(json.loads(path.read_text()),dict(version=3))


if __name__=='__main__':unittest.main()
