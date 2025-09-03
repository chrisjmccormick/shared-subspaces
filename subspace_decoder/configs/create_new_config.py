#!/usr/bin/env python3
"""Utility for creating new JSON config files from an existing one.

The script copies an existing config file and applies overrides passed on the
command line. This repository uses brittle JSON configs and will raise an error
if you try to modify a key that does not already exist.

Required Arguments:
    name (str): The base name for the new config file (e.g., "my_experiment").
                This is used for the output filename and the output directory path.
    --base (str): Path to the existing JSON config file to use as a template.
    --shorthand (str): A short, descriptive string for the experiment. This
                       overwrites the 'shorthand' key in the config.
    --notes (str): A longer description of the experiment's purpose. This
                   overwrites the 'notes' key in the config.

Optional Arguments:
    --set (str): Override a specific key-value pair in the config. Can be used
                 multiple times. Uses dot notation for nested keys.
                 Example: --set model.hidden_size=512

Example usage:
    python create_new_config.py my_experiment \
        --base ./bert_baseline.json \
        --shorthand "mlp.1024, model.512.lyr.6, ah.8.32" \
        --notes "BERT baseline with larger model size" \
        --set model.hidden_size=512 \
        --set pre_train.train_batch_size=128

The new config will be written next to the base file as `<name>.json` and
`pre_train.output_dir` will be set to `checkpoints/<name>`.
"""

import argparse
import json
import os
from typing import Any, Dict


def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Assign `value` inside a nested dictionary `cfg` at a given `path`.

    The path is a string with keys separated by dots (e.g., "pre_train.model.size").

    Args:
        cfg: The dictionary to modify.
        path: The dot-separated path to the key.
        value: The new value to assign.

    Raises:
        KeyError: If any key along the path does not exist in the dictionary.
                  This is intentional to prevent creating new keys accidentally.
    """
    # Split the path into individual keys.
    keys = path.split(".")
    # Start traversal from the top level of the dictionary.
    cur = cfg
    # Traverse the dictionary to the second-to-last key.
    for key in keys[:-1]:
        # If a key in the path doesn't exist, raise an error.
        if key not in cur:
            raise KeyError(f"Unknown config path: {path}")
        # Move one level deeper into the dictionary.
        cur = cur[key]
    
    # Get the final key for the assignment.
    final_key = keys[-1]
    # Check if the final key exists before attempting to assign the value.
    if final_key not in cur:
        raise KeyError(f"Unknown config path: {path}")
    
    # Assign the new value to the final key.
    cur[final_key] = value


def main() -> None:
    # Set up the command-line argument parser, using the script's docstring as a help message.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    # === Define Command-Line Arguments ===

    # A required positional argument for the experiment's name.
    parser.add_argument("name", help="Base name for the new config file")
    
    # A required argument pointing to the base config file.
    parser.add_argument(
        "--base",
        required=True,
        help="Existing config file to copy",
    )
    # A required argument for a short description of the experiment.
    parser.add_argument(
        "--shorthand",
        required=True,
        help="Short descriptor for the experiment",
    )
    # A required argument for longer notes about the experiment.
    parser.add_argument(
        "--notes",
        required=True,
        help="Notes describing the run",
    )
    # An optional, repeatable argument to override specific config values.
    parser.add_argument(
        "--set",
        action="append",  # Allows this argument to be specified multiple times.
        default=[],       # If not used, defaults to an empty list.
        metavar="path=value",
        help="Override a configuration value using dotted paths (e.g., model.hidden_size=256)",
    )
    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # --- Load and Modify Configuration ---

    # Open and load the base JSON configuration file into a Python dictionary.
    with open(args.base) as f:
        config = json.load(f)

    # Directly update the 'shorthand' and 'notes' fields with the provided arguments.
    config["shorthand"] = args.shorthand
    config["notes"] = args.notes

    # The 'pre_train.output_dir' is a special field that is always updated.
    # First, assert that the key path exists in the base config to avoid errors.
    assert "pre_train" in config and "output_dir" in config["pre_train"], (
        "Base config missing required key 'pre_train.output_dir'"
    )
    # Update the output directory to follow the convention 'checkpoints/<experiment_name>'.
    config["pre_train"]["output_dir"] = f"checkpoints/{args.name}"

    # Process all user-specified overrides from the '--set' arguments.
    for item in args.set:
        # Each '--set' item must be in the format "path.to.key=value".
        try:
            # Split the item into path and value at the first equals sign.
            path, raw_value = item.split("=", 1)
        except ValueError as e:
            # If split fails, the format is invalid.
            raise ValueError(f"Could not parse --set '{item}': Must be in 'path=value' format. {e}")
        
        # The value part of the string must be valid JSON. This allows setting
        # numbers, booleans, null, lists, and objects, not just strings.
        try:
            # Use json.loads() to parse the value string into a Python object.
            value = json.loads(raw_value)
        except json.JSONDecodeError as e:
            # If parsing fails, the value is not valid JSON.
            # Note: For a raw string value, it must be enclosed in quotes, e.g., --set model.name='"MyModel"'
            raise ValueError(f"Value for '{path}' is not valid JSON: {e}")
        
        # Use the helper function to set the parsed value at the specified nested path.
        _set_nested(config, path, value)

    # --- Write New Configuration File ---

    # Determine the directory of the base config file.
    base_dir = os.path.dirname(os.path.abspath(args.base))
    # Define the path for the new config file, placing it in the same directory.
    out_path = os.path.join(base_dir, f"{args.name}.json")

    # Write the modified dictionary to the new JSON file.
    with open(out_path, "w") as f:
        # Use indent=2 for a human-readable, pretty-printed format.
        json.dump(config, f, indent=2)
        # Add a trailing newline for POSIX compatibility.
        f.write("\n")
    
    print(f"Wrote new config to {out_path}")


if __name__ == "__main__":
    main()