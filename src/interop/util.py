import string
import random

from onnx.backend.base import namedtupledict

def generate_random_filename(length: int = 24, extension: str = "") -> str:
    """Generates a random filename"""
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    if extension:
        if "." in extension:
            pieces = extension.split(".")
            last_extension = pieces[-1]
            extension = last_extension
        return f"{random_string}.{extension}"
    return random_string

def convert_dictionary_to_namedtupledict(dict, name:str):
    keys = []
    values = []
    for kv in dict:
        keys.append(kv.Key)
        values.append(kv.Value)
    return namedtupledict(name, keys)(*values)