import pytest

from ramlab.state.polarisation import Polarisation

def test_polarisation():
    perp = Polarisation.PERPENDICULAR
    par = Polarisation.PARALLEL
    both = perp | par

    assert perp == Polarisation.PERPENDICULAR
    assert par == Polarisation.PARALLEL
    assert perp != par
    assert both == Polarisation.PERPENDICULAR | Polarisation.PARALLEL
    assert both != perp
    assert both != par

    assert Polarisation.from_string("=") == Polarisation.PARALLEL
    assert Polarisation.from_string("+") == Polarisation.PERPENDICULAR

    with pytest.raises(ValueError):
        Polarisation.from_string("_")