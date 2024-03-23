import string
import random

import numpy as np

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