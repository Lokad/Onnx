## About
This project contains Python modules for interoperating with the .NET tensors library and ONNX backend. The main purpose is to facilitate writing
pytest unit tests that compare ONNX operations in .NET with their Python and Numpy equivalents.

## Usage
1. Create a Python venv e.g. `python -m venv lokadonnx`. (optional)
2. Install the Python dependencies: `pip install -r src\interop\requirements.txt`
3. Build and publish the Lokad.Onnx.Interop project by running the project build script from the repository root folder e.g. `build.cmd`.
3. Run `pytest` from the repository root folder to run the [Python test suite](../../tests/python).
