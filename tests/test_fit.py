import pytest


from Examples.model import *


def test_residual():
    pass


def test_run_model():
    # breakpoint()
    pass


def test_recapitulate_main():
    
    pop = get_population("./tests/data/AustinFacts.csv")
    pars = acquire_params("./tests/data/params_pop_sizes.json")

    float_keys = ['beta', 'mu', 'sigma', 'gamma', 'omega']
    int_keys = ['start_S', 'start_E', 'start_I', 'start_R', 'duration']
    str_keys = ['outdir']
    #validate_params(pars, float_keys, int_keys, str_keys)

    # ----- Run model if inputs are valid -----#

    seir_model = SEIR(**pars)
    # r = seir_model.integrate()
    # seir_model.plot(r)
