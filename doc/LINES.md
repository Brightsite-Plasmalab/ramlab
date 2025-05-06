# Line list calculation

## API
The Python API for calculating and accessing the line lists has the following methods:

```python
Molecule.E(state)
Molecule.dE(state_initial, state_final)
Molecule.degeneracy(state)
Molecule.crosssection_perpendicular(
        state_initial, state_final, lambda_laser
    )
Molecule.crosssection_parallel(state_initial, state_final, lambda_laser)
```

## Saving format
The line lists calculated with RamLab are in a modified [HITRAN2004](https://lweb.cfa.harvard.edu/HITRAN/formats.html) format as displayed below.

Changes:
- Change "Line intensity" to "cross-section"
- Change "R^2" to "Depolarization ratio"
- change (upper -> lower) to (initial -> final) to support anti-Stokes scattering.
- remove parameters that are not appliccable to Raman scattering.


|Symbol|Parameter|Field length|Field start|Data type|Comments or units|
|--- |--- |--- |--- |--- |--- |
|M|molecule number|2|0|Integer|HITRAN chronological assignment|
|I|isotopologue number|1|2|Integer|Ordering by terrestrial abundance|
|ν|Vacuum wavenumber|12|3|Real|cm−1|
|S|Cross-section|10|15|Real| |
|$\rho$|Depolarization ratio|10|25|Real|-|
|||5|35||*unused*|
|||5|40||*unused*|
|E″|Lower-state energy|10|45|Real|cm−1|
|||4|55||*unused*|
|||8|59||*unused*|
|V′|Upper-state “global” quanta|15|67|Text|see Table 3 of Ref. [14]|
|V″|Lower-state “global” quanta|15|82|Text|see Table 3 of Ref. [14]|
|q′|Upper-state “local” quanta|15|97|Text|see Table 4 of Ref. [14]|
|Q″|Lower-state “local” quanta|15|112|Text|see Table 4 of Ref. [14]|
|||6|127||*unused*|
|||12|133||*unused*|
|*|Flag|1|145|Text|Pointer to program and data for the case of line mixing|
|g′|Statistical weight of the upper state|7|146|Real|See details in Ref. [15]|
|g″|Statistical weight of the lower state|7|153|Real|See details in Ref. [15]|

For the quantum numbers, see [the HITRAN specification](https://www.sciencedirect.com/science/article/pii/S0022407305001081#tbl3).