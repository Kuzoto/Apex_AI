"""
json2toon.py - Simple JSON to TOON converter using python-toon

A lightweight wrapper around the python-toon library for easy conversion
between JSON and TOON formats in Python scripts.

Usage:
    from json2toon import json_to_toon, toon_to_json, convert_file
    
    # Convert Python dict/list to TOON string
    toon_str = json_to_toon({"name": "Alice", "age": 30})
    
    # Convert TOON string back to Python dict/list
    data = toon_to_json(toon_str)
    
    # Convert files
    convert_file("input.json", "output.toon")
"""

import json
from typing import Any, Optional, Dict
from pathlib import Path

try:
    from toon import encode, decode
except ImportError:
    raise ImportError(
        "python-toon is not installed. Install it with: pip install python-toon"
    )


def json_to_toon(
    data: Any,
    indent: int = 2,
    delimiter: str = ",",
    length_marker: bool = False
) -> str:
    """
    Convert Python data (dict/list/primitives) to TOON format string.
    
    Args:
        data: Python data structure (dict, list, or primitive)
        indent: Spaces per indentation level (default: 2)
        delimiter: Array delimiter: "," (comma), "\t" (tab), or "|" (pipe)
        length_marker: Add # prefix to array lengths (default: False)
    
    Returns:
        TOON-formatted string
    
    Example:
        >>> data = {"name": "Alice", "age": 30}
        >>> print(json_to_toon(data))
        name: Alice
        age: 30
    """
    options = {
        "indent": indent,
        "delimiter": delimiter,
    }
    
    if length_marker:
        options["lengthMarker"] = "#"
    
    return encode(data, options)


def toon_to_json(
    toon_str: str,
    indent: int = 2,
    strict: bool = True
) -> Any:
    """
    Convert TOON format string back to Python data structure.
    
    Args:
        toon_str: TOON-formatted string
        indent: Expected number of spaces per indentation level (default: 2)
        strict: Enable strict validation (default: True)
    
    Returns:
        Python data structure (dict, list, or primitive)
    
    Example:
        >>> toon_str = "name: Alice\\nage: 30"
        >>> data = toon_to_json(toon_str)
        >>> print(data)
        {'name': 'Alice', 'age': 30}
    """
    try:
        from toon import DecodeOptions
        options = DecodeOptions(indent=indent, strict=strict)
        return decode(toon_str, options)
    except ImportError:
        # Fallback if DecodeOptions is not available
        return decode(toon_str, {"indent": indent, "strict": strict})


def json_string_to_toon(
    json_str: str,
    indent: int = 2,
    delimiter: str = ",",
    length_marker: bool = False
) -> str:
    """
    Convert JSON string to TOON format string.
    
    Args:
        json_str: JSON-formatted string
        indent: Spaces per indentation level (default: 2)
        delimiter: Array delimiter: "," (comma), "\t" (tab), or "|" (pipe)
        length_marker: Add # prefix to array lengths (default: False)
    
    Returns:
        TOON-formatted string
    
    Example:
        >>> json_str = '{"name": "Alice", "age": 30}'
        >>> print(json_string_to_toon(json_str))
        name: Alice
        age: 30
    """
    data = json.loads(json_str)
    return json_to_toon(data, indent, delimiter, length_marker)


def toon_to_json_string(
    toon_str: str,
    indent: Optional[int] = None,
    strict: bool = True
) -> str:
    """
    Convert TOON format string to JSON string.
    
    Args:
        toon_str: TOON-formatted string
        indent: JSON indentation (None for compact, int for pretty-print)
        strict: Enable strict TOON validation (default: True)
    
    Returns:
        JSON-formatted string
    
    Example:
        >>> toon_str = "name: Alice\\nage: 30"
        >>> print(toon_to_json_string(toon_str, indent=2))
        {
          "name": "Alice",
          "age": 30
        }
    """
    data = toon_to_json(toon_str, strict=strict)
    return json.dumps(data, indent=indent)


def convert_file(
    input_path: str,
    output_path: str,
    indent: int = 2,
    delimiter: str = ",",
    length_marker: bool = False,
    strict: bool = True
) -> None:
    """
    Convert between JSON and TOON file formats.
    
    Automatically detects format based on file extension:
    - .json → .toon (encode)
    - .toon → .json (decode)
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        indent: Indentation spaces (default: 2)
        delimiter: Array delimiter for TOON encoding (default: ",")
        length_marker: Add # to array lengths when encoding (default: False)
        strict: Strict validation when decoding TOON (default: True)
    
    Example:
        >>> convert_file("data.json", "data.toon")
        >>> convert_file("data.toon", "output.json")
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Determine conversion direction
    if input_file.suffix.lower() == '.json':
        # JSON to TOON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        toon_str = json_to_toon(data, indent, delimiter, length_marker)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(toon_str)
        
        print(f"Converted {input_path} → {output_path} (JSON → TOON)")
    
    elif input_file.suffix.lower() == '.toon':
        # TOON to JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            toon_str = f.read()
        
        data = toon_to_json(toon_str, indent, strict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        
        print(f"Converted {input_path} → {output_path} (TOON → JSON)")
    
    else:
        raise ValueError(
            f"Unsupported input format: {input_file.suffix}. "
            "Expected .json or .toon"
        )


# Example usage
if __name__ == "__main__":
    # Example 1: Simple conversion
    print("=" * 50)
    print("Example 1: Simple object")
    print("=" * 50)
    
    data = {"name": "Alice", "age": 30, "city": "New York"}
    toon_str = json_to_toon(data)
    print("TOON format:")
    print(toon_str)
    print()
    
    # Convert back
    recovered = toon_to_json(toon_str)
    print("Recovered data:", recovered)
    print()
    
    # Example 2: Array with tabular format
    print("=" * 50)
    print("Example 2: Tabular array")
    print("=" * 50)
    
    users = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 3, "name": "Charlie", "age": 35},
    ]
    
    toon_str = json_to_toon(users)
    print("TOON format:")
    print(toon_str)
    print()
    
    # Example 3: Complex nested structure
    print("=" * 50)
    print("Example 3: Nested structure with length markers")
    print("=" * 50)
    
    complex_data = {
        "metadata": {"version": 1, "author": "test"},
        "items": [
            {"id": 1, "name": "Item1"},
            {"id": 2, "name": "Item2"},
        ],
        "tags": ["alpha", "beta", "gamma"],
    }
    
    toon_str = json_to_toon(complex_data, length_marker=True)
    print("TOON format with length markers:")
    print(toon_str)
    print()
    
    # Example 4: JSON string conversion
    print("=" * 50)
    print("Example 4: JSON string to TOON")
    print("=" * 50)
    
    json_str = '{"name": "Ada", "languages": ["Python", "JavaScript", "Rust"]}'
    toon_str = json_string_to_toon(json_str)
    print("TOON format:")
    print(toon_str)