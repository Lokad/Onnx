# Isolated Winograd product source

M33 component closure 302de6c9 admits product qualification. This source snapshot
adds the exact two qualified Winograd files, optional immutable graph-owned
weights and stride-one dispatch. Direct weights and fallback remain. All direct
candidates are packed before optional clones; both arrays count against shared
budgets. Runtime refusal returns scratch before selected direct fallback.

Only packing, dispatch and focused backend tests differ from qualified root
fe4eb657. Root remains unchanged. Existing numerical comparisons use the
prospective 1e-4 bound only for eligible changed arithmetic; exact fallback,
repeat, held-output and operand checks remain. New tests cover shared budgets,
independent double convolution, overflow/refusal and recovery.

Run `C:/Python313/python.exe -X utf8 -B tests/pyannote/winograd-product-source-v2/prepare.py`
from the repository root, then the separately frozen AMD build/full qualification
tools. This command creates a fresh source snapshot, diff and identity receipt;
it does not build, dispatch or integrate the root product. The complete native,
model, public, long-meeting and matched application/ORT gates still apply.

The initial normal build compiled but failed strict scope verification because
a new Tensor helper renumbered generated names throughout that partial type.
This second source puts the unchanged helper in ConvWinogradDispatch; the
failed first build remains retained, with no numerical or timing claim.
