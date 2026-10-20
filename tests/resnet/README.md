# ResNet full-output reference

The backend tests compare every output component for the constant and ramp
inputs against `fixtures/resnet-ort.json`. The fixture is generated solely by
ONNX Runtime 1.29.0 on CPU, with sequential execution, one thread and all native
graph optimizations enabled. The generator checks the local model hash and
three bit-identical native repeats; it performs no downloads.

From the repository root, in a Python environment with NumPy and ORT 1.29.0:

    python tests/resnet/generate_reference.py --output artifacts/resnet-reference.json

Compare the generated fixture with the committed one before updating it.
The test model remains ignored under `models/resnet50-onnx/model.onnx`.

The former FMA-gated hash froze one historical managed implementation, while
the mean only tied it approximately to ORT. Valid scalar/SIMD/intrinsic and
native convolution paths produce different last bits. The replacement checks
all 2,048 components per case with `abs(actual-reference) <= 1e-5 *
max(1, abs(reference))`, retaining the existing mean/spot checks. Measured
baseline absolute errors against native outputs are at most 4.18e-6. A
separate exact repeatability assertion checks that another managed execution
preserves every bit and the previously returned output.
