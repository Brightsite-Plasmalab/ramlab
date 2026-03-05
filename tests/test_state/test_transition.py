import numpy as np
import pytest

from ramlab.state.transitions2 import Transitions
from ramlab.state.state2 import State

state1 = State(J=[1, 2, 3], v=[0, 1, 0])
state2 = State(J=[1, 3, 4], v=[1, 2, 0])
transition = Transitions(state1, state2)

def test_init():
    Transitions(state1, state2, energies=[0.1, 3, 1])

    state3 = State(J=[1, 2], v=[0, 1])
    with pytest.raises(ValueError):
        Transitions(state1, state3)

    with pytest.raises(ValueError):
        Transitions(state1, state2, energies=[0.1, 3])

def test_to_states():
    s1, s2 = transition.to_states()
    assert s1.equivalent(state1)
    assert s2.equivalent(state2)

def test_attributes():
    # State attributes
    assert np.array_equal(transition.J_initial, np.array([1, 2, 3]))
    with pytest.raises(AttributeError,
                       match="'J' is a State attribute, not Transition attribute,"
                             " add '_initial' or '_final' to specify which state."):
        transition.J

    # Change attributes
    dJ = transition.dJ
    dv = transition.dv
    assert np.array_equal(dJ, np.array([0, 1, 1]))
    assert np.array_equal(dv, np.array([1, 1, 0]))

    with pytest.raises(AttributeError):
        transition.dJ = [0, 1, 1]

    with pytest.raises(AttributeError):
        transition.dk

    transition.state_final

def test_make_mask():
    trans = Transitions(state1, state2, energies=[0.1, 3, 1])

    mask = trans.make_mask(J_initial=1)
    assert np.array_equal(mask, np.array([True, False, False]))

    mask = trans.make_mask(J_final=1)
    assert np.array_equal(mask, np.array([True, False, False]))

    mask = trans.make_mask(energies=1)
    assert np.array_equal(mask, np.array([False, False, True]))

def test_getitem():
    trans = Transitions(state1, state2, energies=[0.1, 3, 1])
    zeroth = trans[0]
    assert zeroth.state_initial.equivalent(state1[0])
    assert zeroth.state_final.equivalent(state2[0])
    assert np.array_equal(zeroth.energies, np.array([0.1]))

    mask = trans.make_mask(J_initial=1)
    assert np.array_equal(trans[mask].energies, np.array([0.1]))

def test_filter():
    trans = Transitions(state1, state2, energies=[0.1, 3, 1])

    trans_filtered = trans.filter(J_initial=1)
    assert trans_filtered.state_initial.equivalent(state1[0])
    assert trans_filtered.state_final.equivalent(state2[0])
    assert np.array_equal(trans_filtered.energies, np.array([0.1]))

    trans_filtered = trans.filter(energies=3)
    assert trans_filtered.state_initial.equivalent(state1[1])
    assert trans_filtered.state_final.equivalent(state2[1])
    assert np.array_equal(trans_filtered.energies, np.array([3]))

def set_attr():
    trans = Transitions(state1, state2, energies=[0.1, 3, 1])
    trans.J_initial = [2, 3, 4]
    assert np.array_equal(trans.state_initial.J, np.array([2, 3, 4]))

    trans.eneries = [3, 1, 0.1]
    assert np.array_equal(trans.energies, np.array([3, 1, 0.1]))

    with pytest.raises(AttributeError, match="'dJ' is read only"):
        trans.dJ = [0, 1, 1]

def test_unique():
    state_1 = state1 + state1
    state_2 = state2 + state2
    trans = Transitions(state_1, state_2, energies=[0.1, 3, 1, 0.2, 3, 1], spin=[0.1, 3, 1, 0.1, 4, 1])
    unique = trans.unique()
    assert unique.state_initial.equivalent(state1)
    assert unique.state_final.equivalent(state2)

    unique2 = trans.unique(use_properties=True)
    initial_expected = state1 + state1[:2]
    final_expected = state2 + state2[:2]
    assert unique2.state_initial.equivalent(initial_expected)
    assert unique2.state_final.equivalent(final_expected)

    unique2 = trans.unique(use_properties='energies')
    initial_expected = state1 + state1[0]
    final_expected = state2 + state2[0]
    assert unique2.state_initial.equivalent(initial_expected)
    assert unique2.state_final.equivalent(final_expected)

    unique2 = trans.unique(use_properties='spin')
    initial_expected = state1 + state1[1]
    final_expected = state2 + state2[1]
    assert unique2.state_initial.equivalent(initial_expected)
    assert unique2.state_final.equivalent(final_expected)
