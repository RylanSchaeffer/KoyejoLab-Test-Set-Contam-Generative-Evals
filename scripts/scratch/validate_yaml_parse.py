"""Validate that every sweep YAML passed on the command line parses.

python scripts/scratch/validate_yaml_parse.py sweeps/pt_gemma3/*.yaml
"""

import sys

import yaml

for path in sys.argv[1:]:
    with open(path) as f:
        parsed = yaml.safe_load(f)
    assert isinstance(parsed, dict) and "parameters" in parsed, path
    print(
        f"ok: {path} (entity={parsed.get('entity')}, project={parsed.get('project')})"
    )
