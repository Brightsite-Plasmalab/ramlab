from abc import ABC, abstractmethod

import numpy as np
import scipy

from ramlab.state.state import State
from ramlab.state.transitions import Transitions
from ramlab.state.polarisation import Polarisation

from ramlab._type_hints import floatNDArray1D, intNDArray1D

kB_per_mK = scipy.constants.value('Boltzmann constant in inverse meter per kelvin')


class Molecule(ABC):
    @abstractmethod
    def E(self, state: State) -> floatNDArray1D:
        """Returns the energy of a given state.

        Parameters
        ------------
            state: The quantum numbers of the states.

        Returns
        ------------
        np.ndarray
            The energies of the states in cm-1.
        """
        pass

    def dE(self, transitions: Transitions) -> floatNDArray1D:
        """Returns the energy difference between two states.

        Parameters
        ------------
        transitions:
            The transitions to calculate the energy difference for.

        Returns
        ------------
        np.ndarray
            The energies of the transitions in cm-1.
        """
        return self.E(transitions.state_final) - self.E(transitions.state_initial)

    @abstractmethod
    def degeneracy(self, state: State) -> intNDArray1D:
        """Returns the degeneracy of a given state.

        Parameters
        ------------
            **state: The quantum numbers of the state.

        Returns
        ------------
        np.ndarray
            The degeneracies of the state [-].
        """
        pass

    @abstractmethod
    def crosssection(self,
                     transitions: Transitions,
                     laser_wavelength: float,
                     polarisation: Polarisation)\
            -> floatNDArray1D:
        """Returns the cross-section of a transition between two states.

        Parameters
        ----------
        transitions: Transitions
            The transitions.
        laser_wavelength: float
            The wavelength of the laser in nm.
        polarisation: Polarisation
            The polarisation of the laser.

        Returns
        -------
        Quantity
            The cross-sections of the transitions.
        """
        pass

    @abstractmethod
    def depolarization_ratio(self, transitions: Transitions) -> floatNDArray1D:
        """Returns the depolarization ratio of a transition between two states.

        Parameters
        ------------
        transitions: Transitions

        Returns
        ------------
        np.ndarray
            The depolarization ratio of the transition.
        """
        pass

    @abstractmethod
    def all_transitions(self) -> Transitions:
        """Returns all possible transitions for the molecule.

        Returns
        ------------
        Transitions
            The transitions.
        """
        pass

    def partition_sum(self, **temperatures: float) -> float:
        """Returns the partition sum of the molecule.

        Parameters
        ------------
        **temperatures:
            The temperatures in Kelvin.

        Returns
        ------------
        float:
            The partition sum of all the states in the molecule.
        """
        states = self.all_states()
        populations = self._relative_populations(states, **temperatures)
        return np.nansum(populations)


    @abstractmethod
    def all_states(self) -> State:
        """Returns all possible states for the molecule.

        Returns
        ------------
        State
        """
        pass


    def _relative_populations(self, state_initial: State, **temperatures: float) \
            -> floatNDArray1D:
        """Returns the relative populations of a state.

        Parameters
        ----------
        state_initial: State
            The initial state.
        temperatures: float
            The temperatures in Kelvin.
        """
        T = self._single_temperature_kwarg(**temperatures)
        g = self.degeneracy(state_initial)
        E = self.E(state_initial)
        return g * np.exp(-100 * E / (kB_per_mK * T))


    def population(self, state_initial: State, **temperatures) \
            -> floatNDArray1D:
        """Returns the populations of a state.

        Parameters
        ----------
        state_initial: State
            The initial state.
        temperatures: float
            The temperatures in Kelvin.

        Returns
        ------------
        float
            The populations of the state.
        """
        rel_pop = self._relative_populations(state_initial, **temperatures)
        partition_sum = self.partition_sum(**temperatures)
        n = rel_pop / partition_sum
        return n

    @abstractmethod
    def get_intensity_constant(
        self, transitions: Transitions, laser_wavelength: float, polarisation: Polarisation
    ) -> floatNDArray1D:
        """Returns the constant part of the intensity calculation. These involve physical constants and cross-sections, but not the populations.

        Parameters
        ------------
        transitions: Transitions
            The transitions.
        laser_wavelength: float
            The wavelength of the laser in nm.
        polarisation: Polarisation
            The polarisation of the laser.

        Returns
        ------------
        float: The intensity constant of the transition.
        """
        pass

    def get_intensity_variable(self, transitions: Transitions, **temperatures: float) -> floatNDArray1D:
        """Returns the variable part of the intensity calculation. These involve populations, but not physical constants or cross-sections.
        When fitting temperatures, this is the only part that changes. Calculating this separately greatly speeds up the fitting process.

        Parameters
        ----------
        transitions: State
            The initial state.
        temperatures: float
            The temperatures in Kelvin.

        Returns
        ------------
            float: The intensity variable of the transition.
        """
        return self.population(transitions.state_initial, **temperatures)

    def get_intensity(
        self,
        transitions: Transitions,
        laser_wavelength: float = None,
        polarisation: Polarisation = Polarisation.COMBINED,
        **temperatures: float,
    ) -> floatNDArray1D:
        """Returns the intensity of a transition.

        Parameters
        ------------
        transitions: Transitions
            The transitions.
        laser_wavelength: float
            The wavelength of the laser in nm.
        polarisation: Polarisation
            The polarisation of the laser.

        Returns
        ------------
        np.ndarray:
            The intensity of the transition.
        """
        I_c = self.get_intensity_constant(transitions, laser_wavelength, polarisation)
        I_v = self.get_intensity_variable(transitions, **temperatures)
        return I_c * I_v

    @staticmethod
    def _single_temperature_kwarg(**temperatures: float) -> float:
        """Returns the temperature from the keyword arguments.

        This is a helper function to make sure that only one temperature is supplied.

        Parameters
        ---------
        temperatures: float
            The temperatures in Kelvin. Should only contain the key ``T``.

        Raises
        --------
        TypeError: If no temperature is supplied.
        NotImplementedError: If multiple temperatures are supplied.
        """
        if len(temperatures) != 1:
            msg = (
                f"Exactly one temperature should be supplied. "
                f"Calculation using multiple temperatures is not implemented for {Molecule.__name__}."
            )
            if Molecule.__class__ != Molecule:
                msg += f" If wanted, this should be implemented in the subclass."
            raise NotImplementedError(msg)
        if "T" not in temperatures:
            raise TypeError("Missing argument: 'T'")
        return temperatures["T"]

# %%