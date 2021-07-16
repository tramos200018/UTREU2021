#Model imported from https://github.com/UT-Covid/SEIR_Example


import numpy as np
import scipy
from scipy import integrate
import json
import argparse
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import least_squares
import pandas as pd

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
        
        self.N = start_S + start_E + start_I + start_R

        if not os.path.isdir(self.outdir):
            self.outdir = '.'

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
    

    #self gets the coefficient variables, x is betta, and data is case data
    def run_model(self, x, **params):
        # have a parameter file as dict params
        # arr = np.array([-(x*data[2]*data[0]) + (self.omega*data[3]), 
        #                  (x*data[0]*data[2]) - (self.sigma)*data[1], self.sigma*data[1] -  self.gamma*data[2], self.gamma*data[2] - self.omega*data[3]])
        # 
        seir_model = SEIR(**params)
        r = seir_model.integrate()
        #seir_model.plot_model(r)


        # keep
        assert isinstance(cases_per_day, np.ndarray)
        return cases_per_day

    
    #don't know how this would work, do I need to make 4 residuals for each compartment?
    def residuals(self, x, y, data):
        """Calculates the residual error."""
        empirical_data = data
        # call convert function
        return empirical_data - run_model(self, x, **params)

    def integrate(self):

        time = np.arange(0, self.duration, 0.01)
        results = scipy.integrate.odeint(self.seir, self.R, time)

        return results

    def plot(self, data):
        time = np.arange(0, len(data))

        plt.title("Cases Over Time") 
        plt.xlabel("time") 
        plt.ylabel("cases") 
        plt.plot(time, data)
        plt.savefig(os.path.join(self.outdir, 'cases.png'))
        plt.show() 


    def plot_model(self, results):

        time = np.arange(0, len(results[:, 1]))

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(time, results[:, 0])
        axs[0, 0].set_title('S')
        axs[0, 1].plot(time, results[:, 1], 'tab:orange')
        axs[0, 1].set_title('E')
        axs[1, 0].plot(time, results[:, 2], 'tab:green')
        axs[1, 0].set_title('I')
        axs[1, 1].plot(time, results[:, 3], 'tab:red')
        axs[1, 1].set_title('R')

        for ax in axs.flat:
            ax.set(xlabel='Time', ylabel='Population')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.savefig(os.path.join(self.outdir, 'SEIR_Model.png'))
        plt.show()

    def fit_to_data(self):
        ##try to fit data
        x0 = np.array([.9])
        params = dict()
        # call to scipy.optimize.least_squares(fun=self.residual, extra_params=params)

        


#converts case data into numpy array
def convert(filename):
    # First read into csv with pandas
    df = pd.read_csv(filename)

    data = df.to_numpy()



    return data



# Reads in csv file
def get_population(filename):

    population = 0


    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        c = 0
    

    
        for row in csv_reader:
            
            if c<2:
                if line_count == 0:
                    #first row
                    line_count += 1
                        
                else:
                    population = row[2].replace(",","")
                    c+=1
                line_count+=1

    return population



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

    #
    #for key in all_keys:
    #    if key not in param_dict.keys():
    #        raise ValueError('Parameter {} missing from input file.'.format(key))


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


def main(opts):

    # ----- Load and validate parameters -----#

    if opts['mode'] == 'single_run':
            
        pop = get_population(opts['pop_file'])
        pars = acquire_params(opts['paramfile'])


        float_keys = ['beta', 'mu', 'sigma', 'gamma', 'omega']
        int_keys = ['start_S', 'start_E', 'start_I', 'start_R', 'duration']
        str_keys = ['outdir']
        #validate_params(pars, float_keys, int_keys, str_keys)

        # ----- Run model if inputs are valid -----#

        seir_model = SEIR(**pars)
        r = seir_model.integrate()
        #seir_model.plot_model(r)

        data = convert(opts['cases_file'])

        dates = data[:,0]
        new_reported = data[:,3]

        seir_model.plot(new_reported)

'''
    elif opts['mode'] == 'fit':
        # do the fitting workflow
    else:
        raise ValueError(f"invalid mode")

'''

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fit', 'single_run'], default="single_run")
    parser.add_argument('-p', '--paramfile', help='Path to parameters file.',
                        default="./inputs/params_pop_sizes.json")
    parser.add_argument('--pop-file', help='Path to population CSV',
                        default="./data/AustinFacts.csv")
    parser.add_argument('--cases-file', help='Path to cases CSV (Johns Hopkins formatting)', 
                        default="./data/Austin_Travis_County_COVID19_Daily_Counts_(Public_View).csv")




    opts = vars(parser.parse_args())

    main(opts)
