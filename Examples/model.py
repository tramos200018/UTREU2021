#Model imported from t


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
    

    
    
    

    def integrate(self):

        time = np.arange(0, self.duration, 0.01)
        results = scipy.integrate.odeint(self.seir, self.R, time)

        return results

    


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

    


#converts case data into numpy array
def convert(filename):
    # First read into csv with pandas
    df = pd.read_csv(filename)

    data = df.to_numpy()



    return data






# Reads in csv file
def get_population(filename):

    population = 0

    vac = 442847

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        c = 0
    

    
        for row in csv_reader:
            
            if c<1:
                if line_count == 0:
                    #first row
                    line_count += 1
                        
                else:
                    population = row[2].replace(",","")
                    c+=1
                line_count+=1

    return int(population) - vac



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

def run_model(x, filename):

    

    pars = acquire_params(filename)
    # edit pars here

    # pars["beta"] = list(x)
    pars["beta"] = np.array([x])



    seir_model = SEIR(**pars)
    r = seir_model.integrate()
    #seir_model.plot_model(r)
    
    # try plotting here

    #print(r)
    cases_reported = r[:,2]

    return cases_reported

def residuals(x, y, filename):
        """Calculates the residual error."""
        empirical_data = y
        #print(x)
        # call convert function

        ans = (empirical_data - run_model(x, filename))/empirical_data * 100
        #print(ans)
        return ans

def plot(data, outdir):
        time = np.arange(0, len(data))

        plt.title("Cases Over Time") 
        plt.xlabel("time") 
        plt.ylabel("cases") 
        plt.plot(time, data)
        plt.savefig(os.path.join(outdir, 'plot.png'))
        plt.show() 


##over here


def fit_to_data(data, filename):
        ##try to fit data
        x0 = [.1]

        '''
        # DEBUG
        print(f'{residuals} is type {type(residuals)}')
        print(f'{x0} is type {type(x0)}')
        print(f'{data} is type {type(data)}')
        print(f'{filename} is type {type(filename)}')
        '''
        # call to scipy.optimize.least_squares(fun=self.residual, extra_params=params)
        x = scipy.optimize.least_squares(residuals, x0, args = (data, filename))
        
        '''
        # DEBUG
        print(f'{x} is type {type(x)}')
        '''
        
        
        return x

def main(opts):

    # ----- Load and validate parameters -----#

    if opts['mode'] == 'single_run':
            
        

        
        data = convert(opts['cases_file'])

        dates = data[:,0]
        cases_reported = data[:,2]

        print(cases_reported)

        model_ran = run_model(0.00000525, opts['paramfile'])
        print(model_ran)

        #print(ans)

        fit_result = fit_to_data(list(cases_reported), opts['paramfile'])
        assert fit_result.success, f"least_squares failed to converge: {fit_result}"
        #ans_with_fitted_beta = run_model(fit_result.x[0], opts['paramfile'])
        print(fit_result.x)
        optimal_betta = .0000105
        #plot(model_ran, "./outputs")

        #plot(new_reported, "./outputs")



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
