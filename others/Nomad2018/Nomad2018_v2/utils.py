import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf

import itertools


# Function to plot important features
def plot_importances(X_train, sorted_features, sorted_importances):
    """
    Args:
        X_train (nd-array) - feature matrix of shape (number samples, number features)
        sorted_features (list) - feature names (str)
        sorted_importances (list) - feature importances (float)
    Returns:
        matplotlib bar chart of sorted importances
    """
    axis_width = 1.5
    maj_tick_len = 6
    fontsize = 14
    bar_color = 'lightblue'
    align = 'center'
    label = '__nolegend__'
    ax = plt.bar(range(X_train.shape[1]), sorted_importances,
                 color=bar_color, align=align, label=label)
    ax = plt.xticks(range(X_train.shape[1]), sorted_features, rotation=90)
    ax = plt.xlim([-1, X_train.shape[1]])
    ax = plt.ylabel('Average impurity decrease', fontsize=fontsize)
    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width, 
                         which='major', right=True, top=True)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tight_layout()
    return ax


# To evaluate performance of the random forest models
def rmsle(actual, predicted):
    """
    Args:
        actual (1d-array) - array of actual values (float)
        predicted (1d-array) - array of predicted values (float)
    Returns:
        root mean square log error (float)
    """
    return np.sqrt(np.mean(np.power(np.log1p(actual)-np.log1p(predicted), 2)))

def plot_actual_pred(train_actual, train_pred, 
                     test_actual, test_pred,
                     target):
    """
    Args:
        train_actual (1d-array) - actual training values (float)
        train_pred (1d-array) - predicted training values (float)
        test_actual (1d-array) - actual test values (float)
        test_pred (1d-array) - predicted test values (float)
        target (str) - target property
    Returns:
        matplotlib scatter plot of actual vs predicted
    """
    s = 75
    lw = 0
    alpha = 0.2
    train_color = 'orange'
    train_marker = 's'
    test_color = 'blue'
    test_marker = '^'
    axis_width = 1.5
    maj_tick_len = 6
    fontsize = 16
    label = '__nolegend__'
    ax = plt.scatter(train_pred, train_actual,
                     marker=train_marker, color=train_color, s=s, 
                     lw=lw, alpha=alpha, label='train')
    ax = plt.scatter(test_pred, test_actual,
                     marker=test_marker, color=test_color, s=s, 
                     lw=lw, alpha=alpha, label='test')
    ax = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.4)    
    all_vals = list(train_pred) + list(train_actual) + list(test_pred) + list(test_actual)
    full_range = abs(np.max(all_vals) - np.min(all_vals))
    cushion = 0.1
    xmin = np.min(all_vals) - cushion*full_range
    xmax = np.max(all_vals) + cushion*full_range
    ymin = xmin
    ymax = xmax    
    ax = plt.xlim([xmin, xmax])
    ax = plt.ylim([ymin, ymax])
    ax = plt.plot([xmin, xmax], [ymin, ymax], 
                  lw=axis_width, color='black', ls='--', 
                  label='__nolegend__')
    ax = plt.xlabel('predicted ' + target, fontsize=fontsize)
    ax = plt.ylabel('actual ' + target, fontsize=fontsize)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width, 
                         which='major', right=True, top=True)
    return ax


# Main function for RandomForestRegressor
def modelRandomForestRegressor(X, y_E, y_Eg, features, test_size, rstate, n_est, max_depth, df_input):
    # split into training and test for the purposes of this demonstration
    X_train_E, X_test_E, y_train_E, y_test_E = train_test_split(X, y_E, 
                                                                test_size=test_size,
                                                                random_state=rstate)
    X_train_Eg, X_test_Eg, y_train_Eg, y_test_Eg = train_test_split(X, y_Eg, 
                                                                test_size=test_size, 
                                                                random_state=rstate)

    rf_E = RandomForestRegressor(n_estimators=n_est, 
                                 max_depth=max_depth,
                                 random_state=rstate)
    rf_Eg = RandomForestRegressor(n_estimators=n_est, 
                                 max_depth=max_depth,
                                 random_state=rstate)
    # fit to training data
    rf_E.fit(X_train_E, y_train_E)
    rf_Eg.fit(X_train_Eg, y_train_Eg)
    
    # collect ranking of most "important" features for E
    importances_E =  rf_E.feature_importances_
    descending_indices_E = np.argsort(importances_E)[::-1]
    sorted_importances_E = [importances_E[idx] for idx in descending_indices_E]
    sorted_features_E = [features[idx] for idx in descending_indices_E]
    print()
    print ("MODEL FIT:")
    r, c = X_train_E.shape
    print ()
    print ("The number of training data:", r)
    r, c = X_test_E.shape
    print ("The number of test data:", r)
    print ()

    print('most important feature for formation energy is %s' % sorted_features_E[0])
    
    # collect ranking of most "important" features for Eg
    importances_Eg =  rf_Eg.feature_importances_
    descending_indices_Eg = np.argsort(importances_Eg)[::-1]
    sorted_importances_Eg = [importances_Eg[idx] for idx in descending_indices_Eg]
    sorted_features_Eg = [features[idx] for idx in descending_indices_Eg]
    print('most important feature for band gap is %s' % sorted_features_Eg[0])
    print()
    
    fig3 = plt.figure(3, figsize=(11,6))
    ax1 = plt.subplot(121)
    ax1 = plot_importances(X_train_E, sorted_features_E, sorted_importances_E)
    ax1 = plt.legend(['formation energy'], fontsize=14, frameon=False)
    ax2 = plt.subplot(122)
    ax2 = plot_importances(X_train_Eg, sorted_features_Eg, sorted_importances_Eg)
    ax2 = plt.legend(['band gap'], fontsize=14, frameon=False)
    plt.tight_layout()
    fig3.savefig("important_feature.pdf")
    #plt.show()
    #plt.close()
    
    y_train_E_pred = rf_E.predict(X_train_E)
    y_test_E_pred = rf_E.predict(X_test_E)
    E_pred = rf_E.predict(df_input)
    target_E = 'formation energy (eV/atom)'
    print('RMSLE for formation energies = %.3f eV/atom (training) and %.3f eV/atom (test)' 
          % (rmsle(y_train_E, y_train_E_pred),  (rmsle(y_test_E, y_test_E_pred))))

    y_train_Eg_pred = rf_Eg.predict(X_train_Eg)
    y_test_Eg_pred = rf_Eg.predict(X_test_Eg)
    Eg_pred = rf_Eg.predict(df_input)
    target_Eg = 'band gap (eV)'
    print('RMSLE for band gaps = %.3f eV (training) and %.3f eV (test)' 
          % (rmsle(y_train_Eg, y_train_Eg_pred), (rmsle(y_test_Eg, y_test_Eg_pred))))
    print()
    print("PREDICTION")
    print("The predicted formation energy is %.3f eV/atom and predicted band gap is %.3f eV " %(E_pred, Eg_pred) )

    fig4 = plt.figure(4, figsize=(11,5))
    ax1 = plt.subplot(121)
    ax1 = plot_actual_pred(y_train_E, y_train_E_pred,
                           y_test_E, y_test_E_pred,
                           target_E)
    ax2 = plt.subplot(122)
    ax2 = plot_actual_pred(y_train_Eg, y_train_Eg_pred,
                           y_test_Eg, y_test_Eg_pred,
                           target_Eg)
    plt.tight_layout()
    fig4.savefig("fit_model.pdf")
    #plt.show()
    #plt.close()

def get_vol(a, b, c, alpha, beta, gamma):
    """
    Args:
        a (float) - lattice vector 1
        b (float) - lattice vector 2
        c (float) - lattice vector 3
        alpha (float) - lattice angle 1 [radians]
        beta (float) - lattice angle 2 [radians]
        gamma (float) - lattice angle 3 [radians]
    Returns:
        volume (float) of the parallelepiped unit cell
    """
    alphaR = alpha * np.pi/180
    betaR = beta * np.pi/180
    gammaR = gamma * np.pi/180
    return a*b*c*np.sqrt(1 + 2*np.cos(alphaR)*np.cos(betaR)*np.cos(gammaR)
                           - np.cos(alphaR)**2
                           - np.cos(betaR)**2
                           - np.cos(gammaR)**2)


def cell33_to_cell6(cell33):
    a1= cell33[0,:]
    a2= cell33[1,:]
    a3= cell33[2,:]
    a = np.sqrt(np.dot(a1, a1))
    b = np.sqrt(np.dot(a2, a2))
    c = np.sqrt(np.dot(a3, a3))
    alp = np.arccos(np.dot(a2, a3)/(b*c))*180.0/np.pi
    bet = np.arccos(np.dot(a1, a3)/(a*c))*180.0/np.pi
    gam = np.arccos(np.dot(a1, a2)/(a*b))*180.0/np.pi
    return np.array([a,b,c,alp,bet,gam])

def readPOSCAR(file_crystal, features):
    f  = open(file_crystal ,"r")
    print ("reading POSCAR:", file_crystal)
    line1 = f.readline()
    Space_group = float(line1)
    line2 = f.readline()
    line3 = np.asarray(f.readline().split())
    line4 = np.asarray(f.readline().split())
    line5 = np.asarray(f.readline().split())
    cell33 = np.column_stack([line3, line4, line5])
    cell33 = np.array(cell33, dtype=np.float)
    Volume = np.linalg.det(cell33)
    #print (Space_group, Volume)
    cell6 = cell33_to_cell6(cell33)
    #print (cell6)
    line7 = f.readline().split()
    line8 = np.asarray(f.readline().split(), dtype=float)
    Natoms = sum(line8)
    #print (Natoms)
    x_frac = line8/Natoms
    x_Al = 0.0
    x_Ga = 0.0
    x_In = 0.0
    for k in range(len(line7)):
        if (line7[k]=="Al"):
            x_Al = x_frac[k]
        if (line7[k]=="Ga"):
            x_Ga = x_frac[k]
        if (line7[k]=="In"):
            x_In = x_frac[k]
    #print (x_Al, x_Ga, x_In)
    f.close
    # creat data frame
    input_data = np.array([Space_group, Natoms,x_Al, x_Ga, x_In])
    input_data = np.append(input_data, cell6)
    input_df = pd.DataFrame(data= [input_data], columns=['sg', 'Natoms', 'x_Al', 'x_Ga', 'x_In', 'a', 'b', 'c', 'alpha', 'beta', 'gamma' ])
    input_df['vol'] = get_vol(input_df['a'], input_df['b'], input_df['c'],
                          input_df['alpha'], input_df['beta'], input_df['gamma'])
    input_df['atomic_density'] = input_df['Natoms'] / input_df['vol']
    return input_df

