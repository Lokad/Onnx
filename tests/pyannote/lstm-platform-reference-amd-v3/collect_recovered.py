"""Recover collection after disk exhaustion without overwriting remote closure inputs."""
import inspect
import run


source=inspect.getsource(run.collect)
source=source.replace('results.tar.gz','results-recovered.tar.gz')
source=source.replace('collection.stderr','collection-recovered.stderr')
source=source.replace('collection.json','collection-recovered.json')
source=source.replace('collection-transfer.json','collection-transfer-recovered.json')
scope=dict(vars(run));exec(compile(source,'recovered-lstm-platform-collection','exec'),scope)
scope['collect']()
