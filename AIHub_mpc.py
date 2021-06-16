""" Example for AI Hub workshop: Model Predictive control with Casadi.

This code is adopted from Matlab implementation 'AIHub_mpc.m' created by 
Reza Ghabcheloo. 

Module contains class for creating simple MP-Controller and a class for executing very simple moving robot simulation.
In the 'main' section of the module the MPC object and robot object are utilized to run a simulation of the controller.    

(c) 2021 Jukka Yrjänäinen

DISCLAIMER: The code is intended for educational use only.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import casadi as ca 
from casadi import SX, DM 
from scipy.integrate import solve_ivp # needed only for simulating robot in the test code

class MPC:
    def __init__(self, N_horizon = 10, Ts = 0.1, circ_obstacles=[]):
        """Initialize Model Predictive Controller object. 

        Args:
            N_horizon (int, optional): Planning horizon lenght. Defaults to 10.
            Ts (float, optional): Timestep durarion in seconds. Defaults to 0.1.
            circ_obstacles (list, optional): List of tuples of form ((x,y),radius) defining static circular obstacles. Defaults to [].
        """

        self.Ts = Ts  # sampling time
        self.N_horizon = N_horizon # horizon lenght in steps

        # ----------------------------- #
        # Define the objective function #
        # ----------------------------- #
        
        # Robot is modelled with state [x,y,yaw], giving position and heading
        # and control input [v,w] representing linear and angular velocity

        n_states = 3
        n_controls = 2 
        
        # Decision variables for NLP
        U = SX.sym('U', n_controls, N_horizon)  # controls of the whole horizon
        X = SX.sym('X', n_states, N_horizon+1)  # states of the whole horizon

        # To pass initial and goal states to the optimization functions as parameters. 
        # This way, you do not have to build the optimization problem everytime you change the goal or init pose 
        x_init = SX.sym('xi', n_states) 
        x_goal = SX.sym('xg', n_states)

        # Weights for quadratic objectives
        Q = DM([1, 1, 0.01])    # states (x,y, yaw)
        Ru = DM([0.3, 0.01])     # cost on controls (v,w) magnitude
        Rdu = DM([0.1, 0.02])    # cost on controls (v,w) change (acceleration/deceleration), 

        # Quadratic cost: difference to the goal state: (x-xg)^2+(y-yg)^2+(yaw-yawg)^2
        # Note: substracting angles in this simple way (x-xg)^2+(y-yg)^2+(yaw-yawg)^2 is not very good idea.
        # So instead of error_to_goal_pos = Ts*ca.sum1( Q[:2]*(X[:2,:-1] - x_goal[:2])**2) we can do:

        # error of position
        error_to_goal_pos = Ts*ca.sum1( Q[:2]*(X[:2,:-1] - x_goal[:2])**2)

        # error of yaw
        adiff = X[2,:-1] - x_goal[2]
        error_to_goal_yaw =  Ts*Q[2]*ca.atan2( ca.sin(adiff), ca.cos(adiff))**2
        
        # total error to goal state 
        error_to_goal =  error_to_goal_pos + error_to_goal_yaw
        
        # Quadratic cost: minimize the magnitude of controls v^2+w^2
        control_power = Ts*ca.sum1(Ru*U**2)
        
        # Quadratic cost: minimize the change of controls: (v-v_prev)^2 + (w-w_prev)^2
        # Note: alternatively this can also be modeled as hard constraint
        acc = ca.diff(ca.horzcat([0,0], U),1,1) # difference to previous control
        control_changes = ca.sum1(Rdu * acc**2) 
                
        # Total objective to be minimized
        total_cost  = ca.sum2( error_to_goal + control_power + control_changes) 

        # One could add a terminal cost at this point. If the time horizon is short,
        # adding it may be necessary to make the MPC stable.
        # This is by no means optimal, but it gives some idea about its effect:
        # Quadratic cost: (x_last - x_goal)^2 + 3*(y_last - y_goal)^2 
        terminal_cost = ca.sum1( DM([1, 3, 0])*(X[:,-1] - x_goal)**2)
        total_cost += terminal_cost

        # ----------------------------- #
        # Define the constrainst        #
        # ----------------------------- #

        # Equality constrainsts
        # ----------------------------- #

        # Start from given inital state
        start_const = X[:,0] - x_init 

        # Constraints between control, current and future states, that is, constraint due to motion model
        # Define the motion model, time derivative of the states: [v*cos(yaw), v*sin(yaw), w]    
        d_state = ca.vertcat( U[0,:] * ca.cos(X[2,:-1]), U[0,:] * ca.sin(X[2,:-1]), U[1,:])

        # --- START an alternative implementation example ---
        
        # With casadi you can also create functions, and it is also possible to construct artefacts
        # sample by sample instead of using vectors. This can be more intuitive, but more verbose:

        # x = SX.sym('x') 
        # y = SX.sym('y')
        # yaw = SX.sym('yaw')
        # states = ca.vertcat(x, y, yaw) 
        
        # v = SX.sym('v') 
        # w = SX.sym('w') 
        # controls = ca.vertcat(v, w)  
 
        # dState = ca.vertcat(v*ca.cos(yaw), v*ca.sin(yaw),w) 
        # fMotion = ca.Function('fMotion',[states,controls],[dState])
        
        # d_state = ca.horzcat(*[ fMotion( X[:,k],U[:,k]) for k in range(N_horizon)])
        
        # --- END an alternative implementation example ---

        # Predict future state with simple intergration.
        # For more advanced methods, see e.g. "Integrator" in casidi's documentation.
        x_predict = X[:,:-1] + Ts*d_state

        # Create constraint 
        model_const = X[:,1:] - x_predict

        # combine equality constraints
        econst = ca.horzcat(start_const, model_const) 
        econst = ca.reshape(econst, 1, econst.size1()*econst.size2())
        N_eq = econst.size2() # number of equality constraints

        # Obstacle constraints
        # ----------------------------- #

        # Constraint for (static) circular obstacles
        oconst = ca.horzcat( *[ca.sum1((X[:2,:] - c)**2) - r**2for c,r in circ_obstacles])
        N_obs = oconst.size2() # number of obstacle constraints

        # Note: What if the obstacles change location?
        # They need to be added as parameters similar to x_init and x_goal

        # ----------------------------- #
        # Construct NLP solver
        # ----------------------------- #
        
        nlp = {
            'f': total_cost, # objective function
            'x': ca.vertcat(ca.reshape(X,3*(N_horizon+1),1), ca.reshape(U,2*N_horizon,1)), # decision varibles as one vector
            'g': ca.horzcat(econst, oconst), # constraints TODO: rethink row vs column vectors, make consistent
            'p': ca.vertcat(x_init, x_goal), # parameters to pass to the optimization
        }

        # optimization options
        opts = {
            'ipopt.max_iter' : 100,
            'ipopt.print_level' : 0,
            'print_time' : 0,
            'ipopt.acceptable_tol' : 1e-6, # Error tolerance
            'ipopt.acceptable_obj_change_tol' : 1e-4
        }

        # Create the solver, we use ipopt here, but there are other options as well.
        self._nlp_solver = ca.nlpsol('nlp_solver', 'ipopt', nlp, opts) 

        # Set constraints and deciosion variable bounds.
 
        # constraint bounds
        # eqauality + obstacles
        self._lbg = ca.vertcat( ca.repmat(0, N_eq, 1), ca.repmat(0, N_obs, 1))
        self._ubg = ca.vertcat( ca.repmat(0, N_eq, 1), ca.repmat(ca.inf, N_obs, 1)) 

        # decision varible bounds: 
        # state    
        self._x_y_yaw_lb = ca.repmat( [-5, -5, -ca.inf], N_horizon+1, 1)  # lower x, y, yaw 
        self._x_y_yaw_ub = ca.repmat( [5, 5, ca.inf], N_horizon+1, 1)  # Upper x, y, yaw

        # velocity
        self.set_speed_limits(-0.5, 0.5, -1, 1)

    # Note: bounds can be altered before each call to the optimizer.
    # Allthough bound vectors can be modified directly, it is better to provide 
    # a higher level method for it, as shown in exmaple below:         
    def set_speed_limits(self, v_min, v_max, w_min, w_max):
        """Method ses velocity constraints for optimization.

        Args:
            v_min (float): minimum linear velocity
            v_max (float): maximum linear velocity
            w_min (float): minimum angular velocity
            w_max (float): maximum angular velocity
        """
        self._v_w_lb = ca.repmat( [v_min, w_min, ], self.N_horizon, 1)  # lower v, w
        self._v_w_ub = ca.repmat( [v_max, w_max ], self.N_horizon, 1)  # Upper v, w
        self._lbx = ca.vertcat(self._x_y_yaw_lb, self._v_w_lb)
        self._ubx = ca.vertcat(self._x_y_yaw_ub, self._v_w_ub)

    def run(self, X0, U0, x_obs, x_goal):
        """Executes MPC optimization and outputs predicted state and control.

        Args:
            X0 (casadi vector): Initial state values
            U0 (casadi vector): Initial control values
            x_obs (iterable of floats): parameter for optimizer, observed current state
            x_goal (iterable of floats): parameter for optimizer, goal state

        Returns:
            [tuple of vectors]: x_traj_predict, u_traj_predict. Optimization results: predicted state and control.
        """

        # Parameter vector for optimizer
        p  = ca.vertcat(x_obs, x_goal)  

        # Initialize the optimization variables
        x0  = ca.vertcat( ca.reshape(X0, 3*(self.N_horizon+1), 1), ca.reshape(U0, 2*self.N_horizon,1) )
        
        # Solve the nlp to find optimal decision variables: X,U
        sol = self._nlp_solver(x0=x0, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg, p=p);
        
        x_traj_predict = ca.reshape( sol['x'][:3*(self.N_horizon+1) ], 3, self.N_horizon+1).full()  
        u_traj_predict = ca.reshape( sol['x'][ 3*(self.N_horizon+1):], 2, self.N_horizon).full()  

        return x_traj_predict, u_traj_predict

    # -----------------------------------------------------------------------------


class MobileRobot:
    def __init__(self, state=[0,0,0], timestep=0.1):
        """Initialize simple mobile robot object

        Args:
            state (list, optional): Initial state. Defaults to [0,0,0].
            timestep (float, optional): timestep in seconds used in simualtion. Defaults to 0.1.
        """
        self._state = state
        self.ts = timestep

    def measure_state(self):
        """Gives current state of the robot.

        Returns:
            [vector]: state (x, y, yaw)
        """
        # state measurememt noise could be added here
        return self._state 
    
    def drive(self, u):
        """Method to model robot movement for simple simulation.

        Args:
            u (vector of float): Control input as [liner velocity, angular velocity]
 
        Returns:
            [vector of float]: New internal state as [x, y, yaw]
        """
        motion_model = lambda t,x :  [ u[0]*np.cos( x[2]), u[0]*np.sin( x[2]), u[1]]
        sol = solve_ivp(motion_model, [0, self.ts], self._state )
        self._state = sol.y[:,-1]

        # Using the ode solver above allows you to simulate the "actual robot" more
        # accurately, although MPC in this example predicts using simple forward Euler.
        # In a real application, the "solve_ivp" line will be just the real robot receiveing the
        # command and next you will measure the state.


if __name__=='__main__':
    
    # Simulate the use of MPC

    max_time = 20 # maximum simulation time in secods
    Ts = 0.1 # time step duration in seconds
       
    # set simulation environment properties
    slow_speed_zone = [0.5, 1.0] # to demostrate changing bounds dynamically
    circles =  [((0.6,0.6),0.2), ((0.75,1.25),0.15)] # obstacles
    # TODO: Experiment with different obstacles

    # create the controller instance
    # TODO: Experiment with different values of planning horizon
    mpc = MPC(N_horizon=15, Ts=Ts, circ_obstacles=circles)

    x0 = np.r_[0, 0, 0]     # init state x,y,yaw
    xg = np.r_[1.5, 1.5, 0] # goal state

    # create instance of robot
    robot = MobileRobot(x0, Ts)
    x_obs = robot.measure_state() # observation, measured state 

    # initial decision variables
    U0 = DM.zeros(2, mpc.N_horizon);        
    X0 = ca.repmat(x0, 1, mpc.N_horizon+1); 

    t = 0 # simulation time

    data = [] # collecting data for plotting

    # The main MPC simulation loop until
    # the goal is reached or simulation time is exceeded
    while np.linalg.norm(x_obs - xg) > 0.1 and t < max_time:
        
        # alter constraints if needed
        if x_obs[1] > slow_speed_zone[0] and x_obs[1] < slow_speed_zone[1]:  
            mpc.set_speed_limits(-0.2, 0.2, -1, 1)
        else:
            mpc.set_speed_limits(-0.5, 0.5, -1, 1)

        # run controller to obtain optimal decision variables: X,U
        x_traj_predict, u_traj_predict = mpc.run(X0, U0, x_obs, xg)
               
        # get the first control for Ts sec
        u_act = u_traj_predict[:,0] 

        # Apply the control
        robot.drive(u_act)
         
        # measure the state
        x_obs = robot.measure_state()
        
        # use the current solution to initialize the next optimization
        X0 =  ca.horzcat(x_traj_predict[:,1:], x_traj_predict[:,-1])
        U0 =  ca.horzcat(u_traj_predict[:,1:], u_traj_predict[:,-1])

        # collect data for plotting 
        data.append([t,x_obs, u_act, x_traj_predict, u_traj_predict])

        # advance one timestep
        t += Ts

    print('Termination after {:.2} s at state:{:}, delta to goal:{:.3}'.format(t, x_obs, np.linalg.norm(x_obs - xg)))

    # Visualize results

    fig, axs = plt.subplots(2,2, figsize=[10,10], tight_layout=True)
    for ax in axs[:,0]:
        ax.set_aspect(1)
        ax.axis([-0.25, 1.75, -0.25, 1.75])
        ax.grid(True)
        ax.plot(*xg[:2],'go')
        for center,radius in circles:
            ax.add_patch(mpatches.Circle( center, radius, fill=False, ec='r'))
        ax.hlines(slow_speed_zone, -0.25, 1.75, linestyle = 'dashed' )
        ax.text(1.0, np.mean(slow_speed_zone), 'Slow speed zone')

    axs[0,0].set_title('All predicted trajectories (x,y)', fontsize='small')
    axs[1,0].set_title('Realized tarjectory (x,y)', fontsize='small')
    
    for t, x_obs, u_act, xp, up in data:
        axs[0,0].plot(xp[0],xp[1]) # all predicted trajectories
        axs[1,0].plot(x_obs[0], x_obs[1], 'b.') # realized tarjectory

    t_axis = [a[0] for a in data]
    
    axs[0,1].set_title('Control vs. time', fontsize='small')
    axs[0,1].grid(True)
    axs[0,1].plot(t_axis, np.array([a[2] for a in data]))
    axs[0,1].legend(['u','w'])
    
    axs[1,1].set_title('State vs. time', fontsize='small')
    axs[1,1].grid(True)
    axs[1,1].plot(t_axis, np.array([a[1] for a in data]))
    axs[1,1].legend(['x','y','yaw'])

    plt.show() 
