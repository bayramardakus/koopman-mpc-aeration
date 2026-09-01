"""
resultspath.py -- locate an archived or freshly regenerated result file.

The analysis scripts write their JSON beside themselves; the repository keeps
the archived copies of those same files under `results/`.  A reader therefore
has to look in both places, and it has to prefer the working directory, so that
a file you have just regenerated wins over the archived one rather than being
silently ignored.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def find(name):
    """Return the path to `name`, preferring a freshly written copy."""
    for cand in (Path.cwd() / name, HERE / name, RESULTS / name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f'{name} not found in the working directory or in {RESULTS}.\n'
        f'The archived copies ship with the repository; if one is missing, '
        f'regenerate it with the stage that writes it (see the "Reproducing '
        f'the results" section of README.md).')


def load(name):
    """Load `name` as JSON from wherever it is."""
    with open(find(name)) as f:
        return json.load(f)
