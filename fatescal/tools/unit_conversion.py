#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converting between units, e.g. to match observations to modelled data.
"""

import numpy as np


def convert_unit(
        values: np.ndarray, unit_in: str, unit_out: str
) -> np.ndarray:
    '''
    Unit converter.
    '''

    implemented_conversions = [
        # (from unit_in, to unit_out)
        ('mmol H2O m-2 s-1', 'kg H2O m-2 s-1'),  # Used for evapotranspiration
        ('µmol CO2 m-2 s-1', 'kg C m-2 s-1'),  # Used for C fluxes
        ('kg H2O m-2 s-1', 'mmol H2O m-2 s-1'),
        ('kg C m-2 s-1', 'µmol CO2 m-2 s-1'),
        ('kg s-1', 'g d-1'),
    ]

    if (unit_in, unit_out) not in implemented_conversions:
        raise NotImplementedError(
            f"Can only convert the following units: {implemented_conversions}"
        )

    # Water fluxes
    if (unit_in, unit_out) == ('mmol H2O m-2 s-1', 'kg H2O m-2 s-1'):
        return np.asarray(values) * 18.01528 * 1e-3 * 1e-3
    if (unit_in, unit_out) == ('kg H2O m-2 s-1', 'mmol H2O m-2 s-1'):
        return np.asarray(values) / 18.01528 / 1e-3 / 1e-3

    # Carbon fluxes
    if (unit_in, unit_out) == ('µmol CO2 m-2 s-1', 'kg C m-2 s-1'):
        return np.asarray(values) * 0.012011 * 1e-06
    if (unit_in, unit_out) == ('kg C m-2 s-1', 'µmol CO2 m-2 s-1'):
        return np.asarray(values) / 0.012011 / 1e-06

    if (unit_in, unit_out) == ('kg s-1', 'g d-1'):
        # X*60(sec->minutes)*60(minutes->hours)*24(h->days)*1000(kg->g)
        return np.asarray(values) * 60 * 60 * 24 * 1000
