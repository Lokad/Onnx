"""Use the complete normal-source audit after the explicit helper correction."""
import importlib.util
from complete import BASE, PREVIOUS, shared, priors


if __name__ == '__main__':
    priors()
    spec = importlib.util.spec_from_file_location('normal_source_v3_audit', PREVIOUS/'audit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()
