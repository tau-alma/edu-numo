# Casadi examples

 Exmaples of using [Casadi](https://web.casadi.org/) to solve optimization problems. Created for AI Hub Tampere Workshop on Numerical Optimal Control (June 18th 2021).

 Disclaimer: Material in this repository is intended for educational use only. Examples are simplified and not suitable for production.

## Installation

Code has been tested in Windows and Linux, and in theory it should also work with Mac. However, some users have reporter issues when installing Casadi on MacOS and manual installation of some libraries might be needed, please consult instructions on [Casadi](https://web.casadi.org/) site. 

Clone this repository to some folder in your machine

    cd my_aihub_stuff
    git clone https://github.com/tau-alma/edu-numo.git
    cd edu-numo 

It is recommended to create a new python virtual envinronment for this. Both [venv](https://docs.python.org/3/library/venv.html) and [conda](https://docs.conda.io/en/latest/) are working. Use relatively new (3.6+) python version. 

In addition to Casadi library, [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/) and [scipy](https://www.scipy.org/) are required dependences.
PLease install those into the virtual envinronment before use. For example:

With venv:

    python -m venv path_to_my_envs/aihub
    path-to-my-envs/aihub/activate
    python -m pip install numpy scipy matplotlib casadi

or if using conda:

    conda create -n aihub -c conda-forge python=3.8 numpy scipy matplotlib casadi
    conda activate aihub

 ## Usage

 There are 3 different examples:

   * AIHub_Circle2rect.py - basic Casadi usage to solve optimization problem.
   * AIHub_trajectory_optimization.py - trajectory optimization with different constraints.
   * AIHub_mpc.py - (simple) model predictive control.

Each example can be launched with command `python filename`, code runs and shows the result of the optimization runs graphically. 

Feel free to experiment with the code examples. For details see comments inside the code. A good hint would be to make a new branch for your experiments, so that you can always conviniently go back to original version.

      git checkout -b myexperiments
