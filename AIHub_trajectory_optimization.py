""" Example for AI Hub workshop: Quadratic programming based trajectory optimization.

This code is adopted from Matlab implementation 'AIHub_TrajOptim.m' created by 
Reza Ghabcheloo. 

(c) 2021 Jukka Yrjänäinen

DISCLAIMER: The code is intended for educational use only.

Problem to be solved:

Given set of way points, generate a smooth trajectory passing close to all of them.

Solution:

1) We reduce the problem to that of finding a polynomial curve of order n_curve. 
2) We defined the desired path to be straght lines connecting the way-points.
3) We build a cost that is quadratic so it is fast to calculate.
     - The cost includes a term for smoothing (square of jerk), and how close the
       trajectory is to the line segments connecting the way points (desired path)
4) different constraints can be added: 
     - velocities and acceleration smooth,
     - close to current pose of the vehicle,
     - start with current speed of the vehicle,
     - pass through a designated area, 
     - etc. other constraints can be added as long as they are quadratic
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import casadi as ca # never use "from casidi import *"
from casadi import SX, DM # some often used unique symbols can be imported into module's namespace, use with caution.

# define the target trajectory as waypoints
roadAhead_x = DM(range(11))
roadAhead_y = DM([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.5, 2.])
NN = roadAhead_x.size1()

# discreate time intervals
tau = np.linspace(0,1,NN)

# Coeffients of the polynomial as casadi vectors
n_curve = 6+1 # order of the polynomials + 1
cx = SX.sym('cx',n_curve,1);  
cy = SX.sym('cy',n_curve,1);

# Polynomial kernels and derivates for each time point
# used for creating objective function for optimization.
pos =  DM([       [               t**n     for n in range(  n_curve)] for t in tau]).T
vel =  DM([   [0]+[             n*t**(n-1) for n in range(1,n_curve)] for t in tau]).T # not used in this example
acc =  DM([ 2*[0]+[       (n-1)*n*t**(n-2) for n in range(2,n_curve)] for t in tau]).T # not used in this example
jerk = DM([ 3*[0]+[ (n-2)*(n-1)*n*t**(n-3) for n in range(3,n_curve)] for t in tau]).T

# objective functions:

# minmize jerk for smoother path
obj_jerk = ca.sum2( (cx.T@jerk)**2 + (cy.T@jerk)**2 ) 

# minimize distace to waypoints 
obj_tracking = ca.sum2( (cx.T@pos - roadAhead_x.T)**2 + (cy.T@pos - roadAhead_y.T)**2) 

# relative weight of smoothness and tracking: larger w, more weight is given to smoothness
# TODO: Experiment with different values of w, start by setting it to 0
w = 1e-5 

nlp = { 'x':ca.vertcat(cx,cy), # optimization variables
        'f': w*obj_jerk + obj_tracking } # objective function

# create solver
S = ca.nlpsol('S', 'ipopt', nlp);
print(S)

# initial guess
x0 = DM.zeros(2*n_curve)

# optimize, no constraints
opt_sol = S(x0=x0)

# -----------------------------------------------------------------------------
# Helper function for showing results - this is just to make later example code shorter
def show_opt_path(sol, ax, title, color='c', n_curve=n_curve, N=100, rx=roadAhead_x, ry=roadAhead_y):

    # get optimized coefficients 
    opt_cx = sol['x'][:n_curve].full()
    opt_cy = sol['x'][n_curve:].full()

    # create interpolated trajectory at 100 points using optimized polynomial
    ipos =  DM([ [t**n for n in range(n_curve)] for t in np.linspace(0,1,100)]).T
    x_pos = opt_cx.T@ipos
    y_pos = opt_cy.T@ipos

    ax.plot(x_pos.T, y_pos.T, color)
    ax.plot(rx,ry,'b.', rx, ry, 'y', lw=0.5) # target waypoints
    ax.set_title(title, fontsize='small')

# set-up plotting
fig, ax = plt.subplots(3,1, figsize=[5,7], tight_layout=True, subplot_kw={'aspect':1, 'ymargin':0.5})

# show results
show_opt_path(opt_sol, ax[0], 'No constraints') # optimised without constraints

# -----------------------------------------------------------------------------
# Adding constraints exmaple: Avoiding circular object
# TODO: Experiment with changing radius or position of the circle.
cir_x, cir_y = (6, 0.5) # center
cir_r = 0.3 # radius

# create position kernels for 50 points in the trajectory
N = 50 
pos =  DM([ [t**n for n in range(n_curve)] for t in np.linspace(0,1,N)]).T
xt = cx.T@pos
yt = cy.T@pos

# create constraint that is positive when position is outside circle
nlp['g'] = (xt-cir_x)**2 + (yt-cir_y)**2 - cir_r**2 

# lower and upperbound for g
lbg = DM.zeros(N,1)  
ubg = DM.inf(N,1)

# create solver
S_obs = ca.nlpsol('S_obs', 'ipopt', nlp)

# optimize with circular object constraint
opt_sol = S_obs(x0=x0, lbg=lbg, ubg=ubg)

# show results
show_opt_path(opt_sol, ax[1], 'Constraint: Object avoidance.')
ax[1].add_patch(mpatches.Circle( (cir_x, cir_y), cir_r, fill=False, ec='r'))

# -----------------------------------------------------------------------------
# Adding constraints example: Start at certain point as equality constraint
# TODO: Experiment by changing the starting point
# starting point constraint
p0=[0, 1.5]

# polynomial kernel for the first point
xt0 = cx.T@pos[:,0]
yt0 = cy.T@pos[:,0]

# constraint is 0 when starting point equals given point
nlp['g'] = ca.horzcat( nlp['g'], xt0-p0[0], yt0-p0[1] )
    
# equal upper and lower bound for equality constraint
lbg = ca.vertcat( lbg, 0, 0)
ubg = ca.vertcat( ubg, 0, 0)

# create solver
S_obs = ca.nlpsol('S_obs', 'ipopt', nlp)

# optimize with circular object constraint and starting point constraint
opt_sol = S_obs(x0=x0, lbg=lbg, ubg=ubg)

# show results
show_opt_path(opt_sol, ax[2], 'Constraints: Starting point and object avoidance.')
ax[2].add_patch(mpatches.Circle( (cir_x, cir_y), cir_r, fill=False, ec='r'))
ax[2].plot(*p0,'r.')

plt.show()