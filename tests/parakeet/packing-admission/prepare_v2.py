"""Retain the old rejection proof and update the existing boundary contract."""
import prepare

prepare.PREDECESSOR=prepare.BASE
prepare.BASE=prepare.ROOT/'artifacts/parakeet-packing-admission-v2-20260921'
if __name__=='__main__':prepare.main()
