"""Preserve the failed collection and fix its missing JOBS import explicitly."""
import inspect
import run


source=inspect.getsource(run.collect)
assert source.count('from protocol import pin,read,save,verify')==1
source=source.replace('from protocol import pin,read,save,verify','from protocol import JOBS,pin,read,save,verify')
source=source.replace('results.tar.gz','results-collected.tar.gz').replace('collection.stderr','collection-results.stderr')
scope=dict(vars(run));exec(compile(source,'corrected-lstm-collection','exec'),scope)
scope['collect']()
