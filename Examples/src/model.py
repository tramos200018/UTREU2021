import numpy as np
import scipy
from scipy import integrate
import xarray as xr
import json
import argparse
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------------------
# -- Read in parameters file
# ----------------------------------------------------------------------------------------------

def acquire_params(filename):

    with open(filename, 'r') as param_file:
        params = json.loads(param_file.read())

    return params

# ----------------------------------------------------------------------------------------------
# -- Validate Inputs
# ----------------------------------------------------------------------------------------------

def validate_params(param_dict, float_keys, int_keys, str_keys):

    all_keys = float_keys + int_keys + str_keys
    for key in all_keys:
        if key not in param_dict.keys():
            raise ValueError('Parameter {} missing from input file.'.format(key))

    for key in float_keys:
        if type(param_dict[key]) != float:
            raise ValueError('Parameter {} is not specified as a float.'.format(key))

    for key in int_keys:
        if type(param_dict[key]) != int:
            raise ValueError('Parameter {} is not specified as an integer.'.format(key))

    for key in str_keys:
        if type(param_dict[key]) != str:
            raise ValueError('Parameter {} is not specified as a string.'.format(key))


# ----------------------------------------------------------------------------------------------
# -- define the system of differential equations
# ----------------------------------------------------------------------------------------------

class SEIR:


    def __init__(self, beta, mu, sigma, gamma, omega, start_S, start_E, start_I, start_R, duration, outdir):

        self.beta = beta  # transmission rate
        self.mu = mu  # death/birth rate
        self.sigma = sigma  # rate E -> I
        self.gamma = gamma  # recovery rate
        self.omega = omega  # waning immunity
        self.start_S = start_S
        self.start_E = start_E
        self.start_I = start_I
        self.start_R = start_R
        self.duration = duration
        self.outdir = outdir
        self.R = [self.start_S, self.start_E, self.start_I, self.start_R]
        
    @property
    def results_xr(self) -> xr.DataArray:
        if hasattr(self, '_results_xr'):
            return self._results_xr
        elif hasattr(self, 'results'):
            assert isinstance(self.results, np.ndarray)
            # convert to DataArray
            self._results_xr = xr.DataArray(
                self.results, dims=('time', 'compartment'),
                coords=dict(compartment=['S', 'E', 'I', 'R'])
            )
            return self._results_xr
        else:
            raise TypeError(f"could not find attribute `results`. Please run " + 
                            "`integrate` method first")

    def seir(self, x, t):

        S = x[0]
        E = x[1]
        I = x[2]
        R = x[3]

        y = np.zeros(4)

        


        y[0] = self.mu - ((((self.beta * I)) * S)) + (self.omega * R) - (self.mu * S)
        y[1] = ((self.beta * S * I)) - (self.mu + self.sigma) * E
        y[2] = (self.sigma * E) - (self.mu + self.gamma) * I
        y[3] = (self.gamma * I) - (self.mu * R) - (self.omega * R)

        return y

    def integrate(self):

        time = np.arange(0, self.duration, 0.01)
        self.results = scipy.integrate.odeint(self.seir, self.R, time)
        return self.results

    def plot_timeseries(self):

        results = self.results

        time = np.arange(0, len(results[:, 1]))

        plt.figure(figsize=(5,8), dpi=300)

        plt.plot(
            time, results[:, 0], "k",
            time, results[:, 1], "g",
            time, results[:, 2], "r",
            time, results[:, 3], "b",)
        plt.legend(("S", "E", "I", "R"), loc=0)
        plt.ylabel("Population Size")
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.title("SEIR Model")
        plt.savefig(os.path.join(self.outdir, 'SEIR_Model.png'))
        plt.show()

    def plot_xr(self):
        """Plot results with different subplots for each compartment. Ref
        https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
        """
        # get compartment labels/coords
        compartments = self.results_xr.coords['compartment'].to_series()

        # number of rows and cols of subplots
        subplot_rows, subplot_cols = 2, 2
        assert subplot_rows * subplot_cols == len(compartments)

        # instantiate a matplotlib plot, with one subplot for each compartment
        fig, axs = plt.subplots(subplot_rows, subplot_cols)
        fig.suptitle('Compartment Population vs Time')

        # set x data
        x = self.results_xr.coords['time']

        # for each compartment, get the y data
        for i, compartment in enumerate(compartments):
            print(f"{i=}, {compartment=}")
            # set y data
            y = self.results_xr.loc[{'compartment': compartment}]
            subplot = axs[i // subplot_rows, i % subplot_cols]
            # plot y vs x
            subplot.plot(x, y)
            # set title of subplot
            subplot.set_title(f'{compartment=}')

        # breakpoint()
        plt.show()

    def plot_example(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(x, y)
        axs[1].plot(x, -y)


def main(opts):

    # ----- Load and validate parameters -----#

    pars = acquire_params("./inputs/params_pop_sizes.json")

    float_keys = ['beta', 'mu', 'sigma', 'gamma', 'omega']
    int_keys = ['start_S', 'start_E', 'start_I', 'start_R', 'duration']
    str_keys = ['outdir']
    validate_params(pars, float_keys, int_keys, str_keys)

    # ----- Run model if inputs are valid -----#

    seir_model = SEIR(**pars)
    seir_model.integrate()
    # seir_model.plot_timeseries()
    seir_model.plot_xr()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paramfile', help='Path to parameters file.')

    opts = parser.parse_args()

    main(opts)