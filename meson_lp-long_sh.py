# -*- coding: utf-8 -*-
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


pd.options.mode.chained_assignment = None  # default='warn'

df_long = df1[df1['proba_long'] >= 0.5]


# Get the top 75 long

sorted_long = df_long.sort_values(['proba_long', 'tradingitemid'], ascending=[0, 0])

sorted75l = sorted_long[0:75]

p_long = sorted75l['proba_long']

w_long = p_long / sum(p_long)

n1= len(p_long)

sorted75l['weight_long'] = w_long

# now the shorts

df_short = df1[df1['proba_short'] >= 0.5]

df_short = df_short[df_short['beta'] >= 0.0]


sorted_short = df_short.sort_values(['proba_short', 'tradingitemid'], ascending=[0, 0])

sorted75s = sorted_short[0:75]

print(sorted75s)

p_sh = sorted75s['proba_short']

w_sh = -p_sh / sum(p_sh)

n2= len(w_sh)

print([len(df_short),n1])

#****************
w_long = pd.Series([1.0/n1 for i in p_long])  # *****temp. hack

w_sh = - w_long

#n1=n2=5

#**************


N = n1 + n2


#************************************
#**********Setting Up the LP **********
#************************************

# Objective function and bounds
c1 = np.concatenate((np.zeros(N),np.ones(N)))

bounds1 = (0,.05)
bounds2 = (-.05,0)

bounds1  = (-.05,.05)


# new constraints for Absolute value  - Check ?

a11 = -1 * np.eye(n1+n2)

a12  = a11

A1= np.concatenate((a11,a12),axis=1)

a21 =  np.eye(n1+n2)

a22  = -a21

A2= np.concatenate((a21,a22),axis=1)


b11 = -w_long[:,None]
b12 = -w_sh[:,None];

b1 =  np.concatenate((b11,b12),axis=0)

b21 = w_long[:,None]
b22 = w_sh[:,None];
b2 =  np.concatenate((b21,b22),axis=0)


#  Exposure Constraint  ***************

M_L   = 1.0
delta_LM  = .1


a31 =  1.0/M_L * np.concatenate((np.ones(n1),np.zeros(n2),np.zeros(n1+n2)))[None,:]

a32  =  -1.0/M_L * np.concatenate((np.ones(n1),np.zeros(n2),np.zeros(n1+n2)))[None,:]


b31  = (1 + delta_LM)
b32  = -(1 - delta_LM)



#*******Shorts

M_S = -1.0
delta_SM = .1

a33 =  1.0/M_S * np.concatenate((np.zeros(n1),np.ones(n2),np.zeros(n1+n2)))[None,:]

a34 =  -1.0/M_S * np.concatenate((np.zeros(n1),np.ones(n2),np.zeros(n1+n2)))[None,:]

b33  = (1 + delta_SM)
b34  = -(1 - delta_SM)

b314 = np.array([b31,b32,b33,b34])[:,None]



#******************



#  Beta Constraint  *********
delta_bM  = 0.1

delta_bS  =  0.6

betal = np.array(sorted75l['beta'])


a41 = np.concatenate((betal,np.zeros(n2),np.zeros(n1+n2)))[None,:]

a42 = -np.concatenate((betal,np.zeros(n2),np.zeros(n1+n2)))[None,:]


b41 = (1 + delta_bM)

b42 = -(1 - delta_bM)

#Shorts
import pdb; pdb.set_trace()
betas = np.array(sorted75s['beta'])

a43 = -np.concatenate((np.zeros(n2),betas,np.zeros(n1+n2)))[None,:]

a44 = np.concatenate((np.zeros(n2),betas,np.zeros(n1+n2)))[None,:]


b43 = (1 + delta_bS)

b44 = -(1 - delta_bS)

b414 = np.array([b41,b42,b43,b44])[:,None]



#*******Shorts



#import pdb; pdb.set_trace()


#  Putting Constraints together ********
#*************
A = np.concatenate((A1,A2,a31,a32,a33,a34,a41,a42,a43,a44))

B  = np.concatenate((b1,b2,b314,b414))

x0 = np.concatenate((w_long, w_sh, np.zeros(150)), axis=0)

#************************************
import pdb; pdb.set_trace()

import scipy

res =scipy.optimize.linprog(c=c1, A_ub=A, b_ub=B, A_eq=None, b_eq=None, bounds=bounds1, method='simplex', callback=None, options={"disp": True})

#import pdb; pdb.set_trace()

print(res.x)

#************************************
# Start the long-short LP
#*********************

