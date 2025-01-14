# Line list generation

This document discusses the classes used in generating line lists.

## State
The `State` class contains information on a quantum-mechanical state of the molecule of interest. It should contain the necessary quantum numbers to describe the state, as well as its energy.

Under the hood, `State` is a wrapper around a simple dictionary. It can be populated using e.g.
```python
state_initial = State(v=[1, 2, 3], J=[4, 5, 6])
```

It supports common operations that can be performed on dictionaries:
```python
# Add/overwrite the v2 quantum number
state_initial["v2"] = [7, 8, 9]

# Read information in a state
print(state_initial["J"])
```

Additionally, it has some features specific to this library:

**Slicing**
```python
state_initial = State(v=[1, 2, 3], J=[4, 5, 6])
state_initial_filtered = state_initial[0:2]

# Slices the values of each property of state_initial
# "State(v, J) with length 2"
```

**Printing**
```python
state_initial = State(v=[1, 2, 3], J=[4, 5, 6])

print(state_initial)
# "State(v, J) with length 3"
```

**Filtering**
```python
state_initial.unique()
# Removes duplicates
```

## Transitions

The `Transitions` class is a wrapper around a pandas `DataFrame`, with some utilities added.
Each row consists of data on the initial and final states, and data on the transition (e.g. energy and cross-section).

**Basic usage**
```python
state_initial = State(J=[1, 2, 3])
state_final = State(J=[3, 4, 5])

transitions = Transitions.for_states(state_initial, state_final)
# Transitions[3](initial_J, final_J)

transitions.dJ = transitions.final_J - transitions.initial_J
# Transitions[3](initial_J, final_J, dJ)

print(transitions.dJ)
# [2 2 2]
```

The `Transitions` class should be instantiated from an initial and final `State` using `Transitions.for_states`.

The opposite can happen as well:
- `Transitions.state_initial` returns a `State` containing every column starting with `initial_`
- `Transitions.state_final` returns a `State` containing every column starting with `final_`

**Sorting**

```python
# Select the 5000 transitions with the highest crosssection
transitions = transitions.sortby("crosssection", ascending=False)[:5000] 
```


## Molecule

The `ramlab.molecule.base.Molecule` (base) class contains a skeleton that every molecule should have implemented;
- `E(state)` gives the energy for a given state.
- `dE(transitions)` gives the energy difference for a transition.
- `degeneracy(state)` returns the degeneracy of a given state.
- `crosssection(transitions, lambda_laser, polarisation)` returns the Raman scattering cross section of a transition.
- `depolarization_ratio(transitions)` returns the depolarization ratio of scattered light for a given transition.
- `get_all_transitions(laser_wavelength, polarisation)` returns all possible transitions for the molecule.

A number of functions are implemented, but can be overridden by specific molecules.
- `get_populations(state_initial, **temperatures)` returns the populations of a state based on a Boltzmann distribution.
- `get_intensity_constant(transitions, laser_wavelength, polarisation)` returns the constant part of the intensity calculation (e.g. unit conversions, cross-sections). These are not re-calculated at every step of a fitting procedure.
- `get_intensity_variable(transitions, **temperatures)` returns the variable part of the intensity calculation (related to temperature-dependent populations). These are recalculated at every step of a fitting procedure.
- `get_intensity(transitions, laser_wavelength, polarisation, **temperatures)` returns the intensity of a transition.

## LinelistMolecule

<!-- TODO: reorganise the library so that LineListMolecule isnt a subclass but a wrapper function? -->
The `ramlab.molecules.hitran_linelist_molecule.LineListMolecule` class is a layer on top of the `Molecule` class that implements saving and loading a line list.

It does this by implementing the `get_all_transitions` function:
- If a line list already exists, load it.
- If it does not exist, calculate it.

Every subclass of `LineListMolecule` then needs to implement how the line list is calculated by implementing `_make_linelist_file`. These linelists should be structured according to the format presented in [LINES.md](./LINES.md). This format contains fields that are equal for every molecule (e.g. wavenumber), and some that are specific to the molecule (e.g. quantum numbers). The latter should be parsed in `process_hitran_data`, such that the correct `Transition` can be created.

## CH4
We discuss the methane class as an example. The methane class (`CH4`) directly inherits from `LinelistMolecule`. 
- It implements `get_linelist_file` to direct to a static file shipped with this library. 
- It implements `process_hitran_data` to process the global and local quanta of the initial and final state, as contained in the HITRAN format.

Implementing another molecule with a line list can be don easily by mirroring the implementation of this `CH4` class.

## AbInitioMolecule

When a line list is not available, cross-sections will be calculated ab initio. This is possible for simple molecules. This project contains monatomic, diatomic and triatomic molecules.

