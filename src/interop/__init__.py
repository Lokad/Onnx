import os
from pythonnet import load, get_runtime_info

load("coreclr")

print(get_runtime_info())