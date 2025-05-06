# Troubleshooting

## Incorrect fits

- Fit at room temperature with a fixed ($T=300K$) temperature.
- Fit a thin spectral line to verify the approximation of the instrument function.
- The instrument has a linespreadfunction in physical space, which approximately translates to wavelength space. Wavenumber space, and especially Raman shift space, are not linearly related to physical space. This means the linespreadfunction will channge depending on spectral position.

A wrong fit can be caused by only a small set of causes:
- Incorrect cross-section
  - incorrect polarizibility
  - incorrect placzek-teller coefficients (check e.g. the Q branch of a vibrational transition, for which the PT coeff are constant)
  - incorrect scaling constants (shouldn't matter if looking at a narrow wavelength region)
- Incorrect population
  - incorrect energy calculation (which would also result in wrong transition wavelengths)
  - incorrect distribution calculation (Boltzmann?)
- Incorrect broadening
  - broadening profile doesn't match reality
  - broadening algorithm broken