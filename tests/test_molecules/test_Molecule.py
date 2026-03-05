import numpy as np
import scipy

from ramlab._type_hints import floatNDArray1D
from ramlab.calculate_molecules import Molecule
from ramlab.state import State, Polarisation, Transitions


class TestState(State):
    _keys = ('rot', 'vib')
    _aliases = {'r': 'rot', 'v': 'vib', 'J': 'rot', 'j': 'rot'}

    def __init__(self, rot, vib):
        super().__init__(rot=rot, vib=vib)

class MoleculeTester(Molecule):
    def E(self, state: State) -> np.ndarray:
        return 10*(state.rot + 1)*state.J + 100*state.vib

    def degeneracy(self, state: State) -> np.ndarray:
        return 1 + 2*(state.rot % 2 != 0)

    def crosssection(self,
                     transitions: Transitions[TestState],
                     laser_wavelength: float,
                     polarisation: Polarisation) -> floatNDArray1D:
        return np.ones_like(transitions.state_initial.rot)

    def depolarization_ratio(self, transitions: Transitions[TestState]) -> floatNDArray1D:
        return np.full_like(transitions.state_initial.vib, 0.5)

    def all_transitions(self) -> Transitions:
        transitions = Transitions.from_state_transition_each(
            self.all_states(),
            {"v": [-1, 0, 1], "J": [-2, 0, 2]}
        )
        mask = transitions.initial_rot >= 0 & transitions.initial_vib >= 0
        return transitions[mask]

    def all_states(self) -> State:
        return TestState.for_each(vib=np.arange(2), rot=np.arange(3))

    def get_intensity_constant(
        self, transitions: Transitions, laser_wavelength: float, polarisation: Polarisation
    ) -> floatNDArray1D:
        return np.ones_like(transitions.initial_rot)

mol = MoleculeTester()
r_val = np.array([1, 2, 3, 1, 2, 3])
vib_val = np.array([0, 0, 1, 0, 0, 1])
r_val2 = np.array([1, 2, 3, 2, 3, 4])
vib_val2 = np.array([1, 1, 0, 0, 0, 1])
test_state1 = TestState(rot=r_val, vib=vib_val)
test_state2 = TestState(rot=r_val2, vib=vib_val2)
transitions = Transitions(test_state1, test_state2)


def Test_E():
    correct_E = 10 * (r_val + 1) * r_val + 100 * vib_val
    assert np.allclose(mol.E(test_state1), correct_E)


def test_dE():
    correct_E1 = 10 * (r_val + 1) * r_val + 100 * vib_val
    correct_E2 = 10 * (r_val2 + 1) * r_val2 + 100 * vib_val2
    correct_dE = correct_E2 - correct_E1

    assert np.allclose(mol.dE(transitions), correct_dE)

def test_degeneracy():
    degen = np.array([3, 1, 3, 3, 1, 3])

    assert np.allclose(mol.degeneracy(test_state1), degen)

def test__relative_populations():
    kB_per_mK = scipy.constants.value('Boltzmann constant in inverse meter per kelvin')
    temperature = 100 / kB_per_mK
    degen = np.array([3, 1, 3, 3, 1, 3])

    correct_E1 = 10 * (r_val + 1) * r_val + 100 * vib_val
    relative_populations = degen * np.exp(-1 * correct_E1)
    assert np.allclose(mol._relative_populations(test_state1, T=temperature), relative_populations)

def test_partition_sum():
    states = mol.all_states()
    kB_per_mK = scipy.constants.value('Boltzmann constant in inverse meter per kelvin')
    temperature = 100 / kB_per_mK
    degen = mol.degeneracy(states)
    energy = mol.E(states)
    relative_populations = degen * np.exp(-1 * energy)
    partition_sum = np.sum(relative_populations)
    assert partition_sum == mol.partition_sum(T=temperature)