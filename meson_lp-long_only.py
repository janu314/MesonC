# -*- coding: utf-8 -*-
'''
created by: Janu314@gmail.com
Edited by: Janu314@gmail.com
last edited: 20171104

'''

import pandas as pd
import graphlab as gl
import numpy as np

df1 = pd.read_csv('ideal_portfolio.2010-01-01.csv')
df1
df1[0:1]
print(len(df1))

df1.columns

'''Index([u'tradingitemid', u'ind_long', u'ind_short', u'proba_long',
       u'proba_short', u'marketcap', u'volume', u'beta', u'borrowcost',
       u'borrowavailability_base'],
      dtype='object')  
    
    tradingitemid  ind_long  ind_short  proba_long  proba_short    marketcap  \
    0        2644491         1          0    0.704813     0.295187  1082.682716
    
    volume      beta  borrowcost  borrowavailability_base
    0  691830.0  0.805392    0.004754               88233850.0 '''


#getting only the longs

#import pdb; pdb.set_trace()

pd.options.mode.chained_assignment = None  # default='warn'

df_long = df1[df1['proba_long'] >= 0.5]


# Get the top 75 long

sorted = df_long.sort_values(['proba_long', 'tradingitemid'], ascending=[0, 0])

sorted75 = sorted[0:75]

p_long = sorted75['proba_long']

w_long = p_long / sum(p_long)

sorted75['weight_long'] = w_long


#import pdb; pdb.set_trace()

n1= len(p_long)

w_long = pd.Series([1.0/n1 for i in p_long])


sorted75[0:2]

#   LP formulations


#import pdb; pdb.set_trace()

# new constraints for Absolute value

a11 = -1 * np.eye(n1)

a12  = a11

A1= np.concatenate((a11,a12),axis=1)

a21 =  np.eye(n1)

a22  = -a21

A2= np.concatenate((a21,a22),axis=1)


b1 =  -w_long[:,None]

b2 =  w_long[:,None]


#import pdb; pdb.set_trace()

#  Exposure Constraint  ***************

M_L   = 1.0
delta_LM  = .1

a31 =  1.0/M_L * np.concatenate((np.ones(n1),np.zeros(n1)))[None,:]


a32  =  -1.0/M_L * np.concatenate((np.ones(n1),np.zeros(n1)))[None,:]



b31  = (1 + delta_LM)
b32  = -(1 - delta_LM)

b312 = np.array([b31,b32])[:,None]




#import pdb; pdb.set_trace()

#  Beta Constraint  *********
delta_bM  = 0.1
delta_cM   = 1.0  # center of Delta

beta = np.array(sorted75['beta'])

a41 = np.concatenate((beta,np.zeros(n1)))[None,:]


a42 = -np.concatenate((beta,np.zeros(n1)))[None,:]


b41 = (delta_cM + delta_bM)

b42 = -(delta_cM - delta_bM)

b412 = np.array([b41,b42])[:,None]



#import pdb; pdb.set_trace()


#************************************
#**********Setting Up the LP **********
#************************************

c1 = np.concatenate((np.zeros(n1),np.ones(n1)))

bounds1 = (0,.05)

A = np.concatenate((A1,A2,a31,a32,a41,a42))

B  = np.concatenate((b1,b2,b312,b412))

#************************************
import pdb; pdb.set_trace()

import scipy

res =scipy.optimize.linprog(c=c1, A_ub=A, b_ub=B, A_eq=None, b_eq=None, bounds=bounds1, method='simplex', callback=None, options={"disp": True})

#import pdb; pdb.set_trace()

print('parameters')
print('Exposure bounds');
print([-b32,b31]);
print('Beta Bounds');
print([-b42,b41]);
print('Weights and Slck Variables')
x0=res.x[0:75]
print(x0)
print[min(x0),max(x0),max(x0)/min(x0)]

sl0  = res.x[75:]
print(sl0)
print[min(sl0),max(sl0),max(sl0)/min(sl0)]

import matplotlib.pyplot as plt
plt.title('Weights and Slacks')
plt.figure(1)
plt.subplot(121)
plt.title('Weights') # subplot 211 title
plt.plot(x0,'bo')

plt.subplot(122)
plt.plot(sl0,'bo')
plt.title('Slacks') # subplot 211 title
#plt.show()

p_exp = np.sum(x0)
p_beta = np.dot(beta,x0)
print('[Exposure,Beta]')
print([p_exp,p_beta])
import pdb; pdb.set_trace()

df_res = pd.DataFrame(x0[None,:])
plt.show()
df_res.to_csv('tmp_res_1025.csv')