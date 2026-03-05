import numpy as np
import pytest

from ramlab.state.state2 import State, RoVibState

state1 = State(J=[1, 2, 3], v=[0, 1, 0])
state2 = State(J=[1, 2, 3], v=[0, 1, 0])
state3 = State(J=[1, 2, 4], v=[0, 1, 0])

def test_init():
    with pytest.raises(ValueError):
        State(J=[1, 2, 3], v=[0, 1])

    with pytest.raises(ValueError):
        State(J="hi", v=[0, 1])

    with pytest.raises(ValueError):
        State(J=['h', 'o'], v=[0, 1])

    with pytest.raises(ValueError):
        State(J=[1, 2, 4], v=[0, 1, 0 + 1j])


def test_len():
    assert len(state1) == 3

def test_add():
    added = state1 + state2
    expected = State(J=[1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0])
    assert added.equivalent(expected)
    state4 = State(J=[1, 2, 4], v=[0, 1, 0], k=[1, 2, 3])

    with pytest.raises(ValueError):
        state1 + state4

def test_append():
    new_state = state1.copy()
    new_state.append(state1)
    expected = State(J=[1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0])
    assert len(new_state) == 6
    assert new_state.equivalent(expected)

    new_state = state1.copy()
    new_state.append(J = [3, 2, 1], v = [9, 2, 4])
    expected = State(J=[1, 2, 3, 3, 2, 1], v=[0, 1, 0, 9, 2, 4])
    assert new_state.equivalent(expected)

    with pytest.raises(ValueError):
        new_state.append(J = [3, 2], v = [9, 2, 1])

    with pytest.raises(ValueError):
        new_state.append(state1, v = [9, 2, 1])

def test_add_each():
    new_state = state1.add_each(k = [19, 3, 1])
    expected = State(J=[1, 2, 3, 1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0, 0, 1, 0], k=[19, 19, 19, 3, 3, 3, 1, 1, 1])
    assert len(new_state) == 9
    assert new_state.equivalent(expected)

def test_transition():
    res = state1.copy()
    initial, final = res.transition(J=np.array([-1, 1]), v=np.array([0, 1]))
    initial_expected = State(J=[1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0])
    final_expected = State(J=[0, 1, 2, 2, 3, 4], v=[0, 1, 0, 1, 2, 1])
    assert len(final) == 6
    assert initial.equivalent(initial_expected)
    assert final.equivalent(final_expected)

    res = state1.copy()
    initial, final = res.transition(J=np.array([1, 2]))
    initial_expected = State(J=[1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0])
    final_expected = State(J=[2, 3, 4, 3, 4, 5], v=[0, 1, 0, 0, 1, 0,])
    assert initial.equivalent(initial_expected)
    assert final.equivalent(final_expected)

def test_transition_each():
    res = state1.copy()
    start, end = res.transition_each(J=np.array([1, 2]))
    start_expected = State(J=[1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0, ])
    end_expected = State(J=[2, 3, 4, 3, 4, 5], v=[0, 1, 0, 0, 1, 0, ])
    assert start.equivalent(start_expected)
    assert end.equivalent(end_expected)

    res = state1.copy()
    start, end = res.transition_each(J=np.array([-2, 1]), v=np.array([0, 1]))
    start_expected = State(J=[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], v=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,])
    end_expected = State(J=[-1, 0, 1, -1, 0, 1, 2, 3, 4, 2, 3, 4], v=[0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1,])
    assert start.equivalent(start_expected)
    assert end.equivalent(end_expected)

def test_unique():
    state = State(J=[3, 2, 1], v=[0, 1, 0])
    added = state + state
    unique = added.unique()
    assert unique.equivalent(state)

def test_setattr():
    test_state = state1.copy()
    assert test_state.equivalent(state1)
    test_state.J = [3, 4, 5]
    assert np.array_equal(test_state.J, np.asarray([3, 4, 5]))
    assert np.array_equal(state1.J, np.asarray([1, 2, 3]))

    with pytest.raises(ValueError):
        test_state.J = [3, 4]

def test_getitem():
    assert np.array_equal(state1["J"], np.array([1, 2, 3]))

    with pytest.raises(KeyError):
        state1["k"]

    assert np.all(state1[[True, True, False]] == State(J=[1, 2], v=[0, 1]))
    with pytest.raises(IndexError):
        state1[[True, True, False, False, True]]

    zeroth = state1[0]
    assert zeroth.equivalent(State(J=[1], v=[0]))

def test_equality():
    assert np.all(state1 == state2)
    assert np.any(~(state1 == state3))
    assert state1.equivalent(state2)

def test_equality_single_multiple():
    multi_state = State(J=np.array([1, 2, 1, 3]), v=np.array([0, 1, 0, 2]))
    single_state = State(J=np.array([1]), v=np.array([0]))
    double_state = State(J=np.array([1, 1]), v=np.array([0, 0]))

    mask = multi_state == single_state
    assert np.array_equal(mask, np.array([True, False, True, False]))
    assert np.all(multi_state[mask] == double_state)
    assert np.all(multi_state[mask] == double_state)

def test_alias():
    rovib = RoVibState(rot=[1, 2, 3], vib=[1, 1, 2])
    assert np.array_equal(rovib.J, np.array([1, 2, 3]))
    assert np.array_equal(rovib.v, np.array([1, 1, 2]))

def test_rovib():
    with pytest.raises(TypeError):
        RoVibState(rot=[1, 2, 3], vib=[1, 1, 1], invalid_key=[1, 2, 3])

def test_for_each():
    rovib = RoVibState.for_each(rot=[1, 2, 3], vib=[1, 1, 2])
    rovib_corr = RoVibState(rot=[1, 2, 3, 1, 2, 3, 1, 2, 3], vib=[1, 1, 1, 1, 1, 1, 2, 2, 2])
    assert rovib.equivalent(rovib_corr)