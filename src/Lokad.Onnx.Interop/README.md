## About
This library contains code to facilitate interop with Python. The corresponding python modules are in the [interop](../interop) folder in the `src` 
directory and can be imported into any Python app. The main rationale is testing correctness of the ONNX backend and ops using the same Python code as the
mainline ONNX source and libraries like Numpy. It's not known how performant the .NET ONNX backend will be from Python due to the cost of interop.

## Guidelines
* Try to have a 1-1 mapping  between the Python module functions and .NET library calls e.g. `get_dims` in the `tensors` module should just map on 
to the `Tensors.GetDims` method in this library as this will reduce the amount of reflection and marshalling needed and increase performance.

* Avoid iteration or repeated calls across language boundaries. Try to do as much as possible in 1 language before passing control to the other.