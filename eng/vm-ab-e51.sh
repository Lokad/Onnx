#!/bin/bash
# E5-1 M3 A/B: fused alpha-twin (F) vs unfused Div-then (U).
# 4 reps ABBA-start per arm, 3 band cases, iters 20, tiering pinned full-opts.
# Tree untouched (env toggle only). Cooldowns separate legs.
set -u
cd $HOME/Onnx || exit 1
export DOTNET_TieredCompilation=0
DLL=$HOME/Onnx/tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll
OUT=$HOME/Onnx/artifacts/ab-e51; mkdir -p $OUT
: > $OUT/order.log
run() {
  arm="$1"; kase="$2"; rep="$3"
  tag="rep${rep}-${arm}-${kase}"
  if [ "$arm" = U ]; then export LOKAD_ONNX_DISABLE_PASSES=scalematmul; else unset LOKAD_ONNX_DISABLE_PASSES; fi
  $HOME/.dotnet/dotnet $DLL e5 --case "$kase" --cpu 2 --iters 20 > "$OUT/${tag}.log" 2>&1
  echo "$tag exit=$? disable=${LOKAD_ONNX_DISABLE_PASSES:-none} tiered=$DOTNET_TieredCompilation" | tee -a $OUT/order.log
}
leg() {
  rep="$1"; a1="$2"; a2="$3"
  for kase in e5-8tok e5-30tok e5-128tok; do
    run "$a1" "$kase" "$rep"; sleep 45
    run "$a2" "$kase" "$rep"; sleep 45
  done
}
leg 1 F U
leg 2 U F
leg 3 U F
leg 4 F U
echo DONE > $OUT/DONE
echo ALLDONE
