# Pinned force fields

CHARMM36 is **not** git-tracked. `python -m tools.md.carb doctor` (or
`prepare`) fetches the MacKerell Lab GROMACS port pinned in
`protocols/amylose_fucoidan/charmm36.pin.json` (URL + sha256) into
`forcefields/charmm36/`.

Checksum mismatch is a loud failure. There is no silent fallback URL.
If the extracted tree is already present, doctor works offline.
