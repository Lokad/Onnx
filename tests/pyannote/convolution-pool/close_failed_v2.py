"""Retain the remaining single static-initializer ordinal mismatch."""
import common
common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v2-20260921'
source = (common.TOOLS / 'close_failed.py').read_text(encoding='utf8')
source = source.replace('Added methods shift compiler-generated ordinals in partial classes; strict method isolation rejected.',
    'All ordinary method ordinals preserved; one generated Tensor static-initializer lambda changes from 528 to 530 because two overloads were added. Strict unnormalized isolation rejected.')
namespace = dict(__name__='static_initializer_failure', __file__=str(common.TOOLS / 'close_failed.py'))
exec(compile(source, namespace['__file__'], 'exec'), namespace)
namespace['main']()
