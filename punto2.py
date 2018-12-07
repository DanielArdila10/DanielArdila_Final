# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 08:41:11 2018

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt



data=np.loadtxt('datos_observacionales.dat')

t_obs=data[:,0]
x=data[:,1]
y=data[:,2]
z=data[:,3]
y_obs=np.array([x,y,z])
x0=x[0]
y0=y[0]
z0=z[0]
sigma_datos_observados=np.ones(len(z))

dt=0.04

def model(param,dt,y_obs):
    sigma=param[0]
    rho=param[1]
    beta=param[2]
 
    x_before=np.array([y_obs[0][0]])
    y_before=np.array([y_obs[0][1]])
    z_before=np.array([y_obs[0][2]])
    
    for i in range(1,31):
        x=x_before[i-1]
        y=y_before[i-1]
        z=z_before[i-1]
        x_next_value=dt*sigma*(y-x)+x
        y_next_value=y+dt*x*(rho-z)-dt*y
        z_next_value=dt*x*y-beta*z*dt+z
        x_before=np.append(x_before,x_next_value)
        y_before=np.append(y_before,y_next_value)
        z_before=np.append(z_before,z_next_value)
        
    ans=np.array([x_before,y_before,z_before])
    return ans



def loglikelihood(t_obs,y_obs,sigma_y_obs, param,dt):
   
    d = y_obs -  model(param,dt,y_obs)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d


def logprior(sigma, beta,rho):
    p = -np.inf
    if sigma < 30 and sigma >0 and beta < 30 and beta >0 and rho < 30 and rho >0:
        p = 0.0
    return p


def divergence_loglikelihood(t_obs, y_obs, sigma_y_obs, param,dt):
    """Divergencia del logaritmo de la funcion de verosimilitud.
    """
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood(t_obs, y_obs, sigma_y_obs, param + delta_parameter,dt) 
        div[i] = div[i] - loglikelihood(t_obs, y_obs, sigma_y_obs, param - delta_parameter,dt)
        div[i] = div[i]/(2.0 * delta)
    return div

def hamiltonian(t_obs, y_obs, sigma_y_obs, param, param_momentum,dt):
    """Hamiltoniano: energia cinetica + potencial: K+V
    """
    m = 100
    K = 0.5 * np.sum(param_momentum**2)/m
    V = -loglikelihood(t_obs, y_obs, sigma_y_obs, param,dt)     
    return K + V

def leapfrog_proposal(t_obs, y_obs, sigma_y_obs, param, param_momentum,dt):
    """Integracion tipo leapfrog. 
        `param` representa las posiciones (i.e. los parametros).
        `param_momemtum` representa el momentum asociado a los parametros.
    """
    N_steps = 5
    delta_t = 1E-2
    m = 100
    new_param = param.copy()
    new_param_momentum = param_momentum.copy()
    for i in range(N_steps):
        new_param_momentum = new_param_momentum + divergence_loglikelihood(t_obs, y_obs, sigma_y_obs, param,dt) * 0.5 * delta_t
        new_param = new_param + (new_param_momentum/m) * delta_t
        new_param_momentum = new_param_momentum + divergence_loglikelihood(t_obs, y_obs, sigma_y_obs, param,dt) * 0.5 * delta_t
    new_param_momentum = -new_param_momentum
    return new_param, new_param_momentum


def monte_carlo(t_obs, y_obs, sigma_y_obs,dt, N=500):
    param = [np.random.random(3)]
    param_momentum = [np.random.normal(size=3)]
    for i in range(1,N):
        propuesta_param, propuesta_param_momentum = leapfrog_proposal(t_obs, y_obs, sigma_y_obs, param[i-1], param_momentum[i-1],dt)
        energy_new = hamiltonian(t_obs, y_obs, sigma_y_obs, propuesta_param, propuesta_param_momentum,dt)
        energy_old = hamiltonian(t_obs, y_obs, sigma_y_obs, param[i-1], param_momentum[i-1],dt)
   
        r = min(1,np.exp(-(energy_new - energy_old)))
        alpha = np.random.random()
        if(alpha<r):
            param.append(propuesta_param)
        else:
            param.append(param[i-1])
        param_momentum.append(np.random.normal(size=3))    

    param = np.array(param)
    return param

param_chain = monte_carlo(t_obs, y_obs, sigma_datos_observados,dt)
n_param  = len(param_chain[0])
best = []
for i in range(n_param):
    best.append(np.mean(param_chain[:,i]))

param_sigma=param_chain[:,0]
param_rho=param_chain[:,1]
param_beta=param_chain[:,2]
t_model = np.linspace(t_obs.min(), t_obs.max(), len(y_obs[0]))
y_model = model(best,dt, y_obs)

plt.figure('X')
plt.errorbar(t_obs,y_obs[0], yerr=sigma_datos_observados, fmt='o', label='obs')
plt.plot(t_model, y_model[0], label='model')

plt.figure('Y')
plt.errorbar(t_obs,y_obs[1], yerr=sigma_datos_observados, fmt='o', label='obs')
plt.plot(t_model, y_model[1], label='model')


plt.figure('Z')
plt.errorbar(t_obs,y_obs[2], yerr=sigma_datos_observados, fmt='o', label='obs')
plt.plot(t_model, y_model[2], label='model')


plt.figure('hist_sigma')
plt.hist(param_sigma)


plt.figure('hist_rho')
plt.hist(param_rho)


plt.figure('hist_beta')
plt.hist(param_beta)


print('faltó colocar los nombres en los ejes y los parámetros no dieron la distribución que tenían que seguir (probablemente el error está en el modelo)')