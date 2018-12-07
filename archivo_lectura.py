# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:34:27 2018

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
'''
data1=np.loadtxt('cadena1.dat')
data2=np.loadtxt('cadena2.dat')
data3=np.loadtxt('cadena3.dat')
data4=np.loadtxt('cadena4.dat')
data5=np.loadtxt('cadena5.dat')
data6=np.loadtxt('cadena6.dat')
data7=np.loadtxt('cadena7.dat')
data8=np.loadtxt('cadena8.dat')
'''

def distribu_analitica(x,sigma,mean):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mean)/sigma)**2)

x=np.linspace(-5,5,100)
result=distribu_analitica(x,1,0)