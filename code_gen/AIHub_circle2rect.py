""" Example for AI Hub workshop: Using casadi-libary to solve non-linear optimization problem.

This code is adopted from Matlab implementation 'AIHubCircle_2_Rectange.m' created by 
Reza Ghabcheloo. 

(c) 2021 Jukka Yrjänäinen

DISCLAIMER: The code is intended for educational use only.

Problem:

Find a distance between a circle and a rectangle using optimization. 

Solution: 

Constraints are, a point inside the circle and another one inside the rectangle.
Find the min distance between these two points.

"""
# Remarks about this implemenation:

# The code follows original Matlab implentation quite closely and similar variable namings are used,
# so that comparing the implementations is easier. If possible, for own implementations use python
# style conventions for variable naming, see https://www.python.org/dev/peps/pep-0008/.

# In addition to obvious syntactic differerences following is important:
#   - Casadi is not imported directly to the namespace of this module. In general avoid
#     'from casidi import *'. Note that casidi's own example documentation uses this import pattern
#     but only to make examples in online documentation shorter, do not copy that pattern to your own
#     code, you will thank me later.    
#   - Unlike Matlab, python does not have inbuilt vector/matrix representation, a standard approach
#     is to use numpy-library. But it is also possible to use native python iterables in many cases
#     with casidi or to use casidi's own DM varibles. The choice depends on the application needs.
#     Casidi's own documentation recommends to use numpy where numerical performance is needed.
  


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import casadi as ca # never use "from casidi import *"
from casadi import SX, DM # some often used symbols can be imported into module's namespace, use with care.  

# Optimization variables
Pcirc = SX.sym('Pcirc',2,1) # point inside the circle
Prect = SX.sym('Prect',2,1) # point inside the rectangle

# Define circle
# TODO: Experiment by changing the position or radius of the circle.
circC = [2, 1]  # circle center
circR = 0.2    # circle radius

# Define 1st constraint:
# A point Pcirc is inside the circle when it's distance form the center
# is smaller than circle radius: g_circle < 0
g_circle = ca.sum1((Pcirc - circC)**2) - circR**2

# Define rectangle
# TODO: Experiment by changing the position or dimension of rectangle.
rectC = [0, 0] # rectangle center
rL, rW =  (0.6/2, 0.4/2) # length, width

# Define 2nd constraint
# A point Prect is inside a rectangle when all g_rect[:] < 0
# Understanding somewhat cryptic implementation details in following lines
# is not very important at the first phase. 
rectLimits =  [ rectC[0]+rL, rectC[0]-rL, rectC[1]+rW, rectC[1]-rW ]
rectA = DM([ [+1,  0], [-1,  0], [0, +1], [0, -1] ])
rectB = ca.vertcat([+1, -1, +1, -1])*rectLimits
g_rect = rectA@Prect - rectB

# Add constraints to the dict that is used for creating optimization function
nlp = {'g': ca.vertcat( g_circle, g_rect) }

print('gsize:', ca.vertcat( g_circle, g_rect).size() )

lbg = -DM.inf(5,1) # lower bounds on g
ubg = DM.zeros(5,1) # upper bounds on g

nlp['x'] = ca.vertcat(Pcirc, Prect) # optimization variables
x0 = ca.vertcat(circC, rectC) # initial guess, centers of circle and rectangle
lbx = -100*np.ones((4,1)) # lower bound on decision variables
ubx = 100*DM.ones(4,1) # upper bound on decision variables

# nlp['f'] = ca.sum1((Pcirc-Prect)**2) # cost function to be optimized

# s = ca.sum1((Pcirc-Prect)**2) # cost function to be optimized
# f = ca.Function('f', [Pcirc, Prect], [s] )
# f.generate('gen.c')

s = ca.Importer('gen.c', 'clang')
nlp['f'] = s

# Construct optimization solver
# dictionary after nlp, defines optimization options, it can be ommitted.
S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt.print_level':0, 'print_time':0})

print(S)

# Solve the oprimization

opt_sol = S(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)



#Optimization results
optPcirc = opt_sol['x'][0:2].full() # Pcirc 
optPrect = opt_sol['x'][2:4].full() # Prect

print(optPcirc, optPrect)

# Display results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis([-0.35, 2.25, -0.3, 1.5])
ax.set_aspect(1)

ax.add_patch(mpatches.Circle( circC, circR, fill=False, ec='r'))
ax.plot(*optPcirc,'go', *circC, 'ro' )
ax.add_patch(mpatches.Rectangle( (rectLimits[1],rectLimits[3]), 2*rL, 2*rW, fill=False, color='b' ))
ax.plot(*optPrect,'co', *rectC, 'bo' )
ax.plot(*zip(optPrect, optPcirc), 'k--' )
plt.show()