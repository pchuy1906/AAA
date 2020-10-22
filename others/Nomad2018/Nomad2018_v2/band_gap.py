#!/usr/bin/env python

print ()
print ("Predict the formation energy and band-gap of semicondoctors with Al, Ga, In")
print ()

import argparse
parser = argparse.ArgumentParser(description='Predict the formation energy and band-gap of semicondoctors with Al, Ga, In')
# Arguments supported by the code.
parser.add_argument("--file_train", default='train.csv', help='training data. Default: train.csv')
parser.add_argument("--test_size", type=float, default=0.2, help='train-test split: test size. Default: 0.2')
parser.add_argument("--rstate", type=int, default=42, help='train-test split: random state. Default: 42')
parser.add_argument("--n_est", type=int, default=100, help='random forest input: n_estimators. Default: 100')
parser.add_argument("--max_depth", type=int, default=5, help='random forest input: max_depth. Default: 5')
parser.add_argument("--file_crystal", default="POSCAR", help='file POSCAR format. Default: POSCAR')

args          = parser.parse_args()
file_train    = args.file_train
test_size     = args.test_size
rstate        = args.rstate
n_est         = args.n_est
max_depth     = args.max_depth
file_crystal  = args.file_crystal

import utils

import pandas as pd

# Load the data and rename the columns

df_data = pd.read_csv(file_train)
df_data = df_data.rename(columns={'spacegroup' : 'sg',
                        'number_of_total_atoms' : 'Natoms',
                        'percent_atom_al' : 'x_Al',
                        'percent_atom_ga' : 'x_Ga',
                        'percent_atom_in' : 'x_In',
                        'lattice_vector_1_ang' : 'a',
                        'lattice_vector_2_ang' : 'b',
                        'lattice_vector_3_ang' : 'c',
                        'lattice_angle_alpha_degree' : 'alpha',
                        'lattice_angle_beta_degree' : 'beta',
                        'lattice_angle_gamma_degree' : 'gamma',
                        'formation_energy_ev_natom' : 'E',
                        'bandgap_energy_ev' : 'Eg'})

# unit-cell volume
df_data['vol'] = utils.get_vol(df_data['a'], df_data['b'], df_data['c'],
                          df_data['alpha'], df_data['beta'], df_data['gamma'])
df_data['atomic_density'] = df_data['Natoms'] / df_data['vol']   

non_features = ['id', 'E', 'Eg']
features = [col for col in list(df_data) if col not in non_features]
print('%i features used in the ML model %s' % (len(features), features))
print()

# Read POSCAR
df_input = utils.readPOSCAR(file_crystal, features)
print ("result after reading POSCAR:")
print (df_input)

# Prepare training data and fit the Random Forest model
X = df_data[features].values
y_E  = df_data['E'].values
y_Eg = df_data['Eg'].values
utils.modelRandomForestRegressor(X, y_E, y_Eg, features, test_size, rstate, n_est, max_depth, df_input)


