"""Human-reference scoring is separate from native application agreement."""
import jiwer
from common import normalize


def distance(left,right):
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def metrics(reference,hypothesis):
    ref,hyp=normalize(reference),normalize(hypothesis)
    if not ref:raise ValueError('Empty normalized human reference')
    words=jiwer.process_words(ref,hyp);chars=jiwer.process_characters(ref,hyp)
    word_errors=distance(ref.split(),hyp.split());char_errors=distance(ref,hyp)
    assert word_errors==words.substitutions+words.deletions+words.insertions
    assert char_errors==chars.substitutions+chars.deletions+chars.insertions
    return dict(reference=ref,hypothesis=hyp,reference_words=len(ref.split()),reference_characters=len(ref),
        word_errors=word_errors,character_errors=char_errors,substitutions=words.substitutions,deletions=words.deletions,insertions=words.insertions,
        word_error_rate=word_errors/len(ref.split()),character_error_rate=char_errors/len(ref),normalized_equal=ref==hyp)


def total(rows):
    keys=('reference_words','reference_characters','word_errors','character_errors','substitutions','deletions','insertions')
    result={k:sum(r[k] for r in rows) for k in keys}
    assert result['word_errors']==result['substitutions']+result['deletions']+result['insertions']
    result.update(word_error_rate=result['word_errors']/result['reference_words'],
                  character_error_rate=result['character_errors']/result['reference_characters'],
                  normalized_exact=sum(r['normalized_equal'] for r in rows))
    return result
