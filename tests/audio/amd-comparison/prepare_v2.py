"""Preserve the initial preparation and build a separate executable payload."""
import prepare

prepare.BASE=prepare.ROOT/'artifacts/audio-amd-comparison-v2-20260920'
prepare.REMOTE='/home/vermorel/Onnx/artifacts/audio-amd-comparison-v2-20260920'

if __name__=='__main__':prepare.main()
