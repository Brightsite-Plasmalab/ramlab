Each molecule added to the RamLab library will be subjected to a number of tests to ensure the simulation accuracy accross different versions of the program.

Requested tests for each molecule:
- Verify energy of a state using HITEMP/HITRAN data
- Partition sum doesn't change much with temperature
- Fit of experimental spectrum
  - Only rotational Stokes
  - Rotational Stokes + rotational anti-Stokes
  - only vibrational Stokes
  - Complete rotational + vibrational and Stokes + anti-Stokes spectrum
- Relative population density < 1
- 