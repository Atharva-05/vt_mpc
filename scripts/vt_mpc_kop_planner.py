#!/usr/bin/env python3

import do_mpc
import do_mpc.data
import traceback
import csv
import numpy as np
from casadi import *
from geometry_msgs.msg import Point
from scipy.interpolate import CubicSpline
import yaml
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class mpcOfflinePlanner():
    
    def __init__(self, visualize: bool):

        self.visualize = visualize

    def plan(self, run_params):
        self.run_params = run_params

        try:
            budget = run_params['budget']
            v_max = run_params['v_max']
            a_max = run_params['a_max']
            dt_min = run_params['dt_min']
            dt_max = run_params['dt_max']
            t_step = run_params['t_step']
            c_b = run_params['c_b']
            r_term = run_params['r_term']
            instance = run_params['instance']
            
            self.initialize_optimizer(budget=budget, v_max=v_max, a_max=a_max, dt_min=dt_min, dt_max=dt_max, t_step=t_step, c_b=c_b, r_term=r_term)
            success = self.plan_full()
            if(success):
                print('Planned trajectory without exceptions.')
            else:
                print('Planning failed.')

        except Exception as e:
            traceback.print_exc()

    def initialize_optimizer(self, budget, v_max, a_max, dt_min, dt_max, t_step, c_b, r_term):
        '''
        Sets up the optimization problem with model, constraint and cost definitions
        '''
        self.solver = 'ma97'                                # IPOPT solver
        self.detection_range = c_b                          # Sensor function parameter -> Half the reward will be collected at this distance from the target
        self.butterworth_order = 2                          # Sensor function parameter
        self.v_max = v_max                     
        self.a_max = a_max
        self.j_max = 30.0                     
        self.max_heading_rate = 0.75
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.r_term = r_term
        self.t_step = t_step
        self.travel_budget = budget
        self.n_horizon = int(self.travel_budget/self.t_step) # prediction horizon of the MPC [timesteps]

        # Load dataset
        self.nodes = []                        
        self.start_node = []
        self.end_node = []
        self.dataset_path = '../dataset/{}.txt'.format(self.run_params['instance'])
        print("Loading file: " + str(self.dataset_path))
        with open(self.dataset_path, newline='') as instance:
            instance_reader = csv.reader(instance, delimiter='\t')
            self.start_node = next(instance_reader)
            self.end_node = next(instance_reader)
            for row in instance_reader:
                self.nodes.append(row)
        self.rewards = np.array([float(node[3]) for node in self.nodes])
        print('Length of dataset: ', len(self.nodes))
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # Defining states and inputs
        awareness = model.set_variable(var_name='awareness', var_type='_x', shape=(len(self.nodes), 1))
        uav_pos = model.set_variable(var_name='uav_pos', var_type='_x', shape=(3, 1))                   # Position
        uav_vel = model.set_variable(var_name='uav_vel', var_type='_x', shape=(3, 1))                   # Linear Velocity
        uav_accel = model.set_variable(var_name='uav_accel', var_type='_x', shape=(3, 1))               # Acceleration
        uav_jerk = model.set_variable(var_name='uav_jerk', var_type='_u', shape=(3, 1))                 # Jerk
        uav_heading = model.set_variable(var_name='uav_heading', var_type='_x', shape=(1, 1))           # Heading
        uav_heading_rate = model.set_variable(var_name='uav_heading_rate', var_type='_u', shape=(1, 1)) # Heading Rate
        t_sum = model.set_variable(var_name='t_sum', var_type='_x')                                     # Total time
        time_dynamic = model.set_variable(var_name='time_dynamic', var_type='_u')                       # Variable timestep
        
        # State update equations
        uav_accel_next = uav_accel + uav_jerk * time_dynamic
        uav_vel_next = uav_vel + (uav_accel * time_dynamic) + ((1/2) * uav_jerk * time_dynamic**2)
        uav_pos_next = uav_pos + (uav_vel * time_dynamic) + ((1/2) * uav_accel * time_dynamic**2) + ((1/6) * uav_jerk * time_dynamic**3)
        uav_heading_next = uav_heading + uav_heading_rate * time_dynamic

        model.set_rhs('uav_accel', uav_accel_next)
        model.set_rhs('uav_vel', uav_vel_next)
        model.set_rhs('uav_pos', uav_pos_next)
        model.set_rhs('uav_heading', uav_heading_next)
        model.set_rhs('t_sum', t_sum + time_dynamic)

        x0 = []
        A_tot = 0
        awareness_next = SX(1,len(self.nodes))
        for k in range(len(self.nodes)):
            x = float(self.nodes[k][0])
            y = float(self.nodes[k][1])
            z = float(self.nodes[k][2])
            x0.append(float(self.nodes[k][3]) + np.random.normal(loc=0, scale=0.1))
            exp = awareness[k] * (1 - 1 / (1 + (sqrt((uav_pos[0] - x)**2 + (uav_pos[1] - y)**2 + (uav_pos[2] - z)**2) / self.detection_range)**self.butterworth_order))
            awareness_next[k] = exp
            A_tot += awareness[k]

        print('Reward values perturbed with noise from normal distribution with standard deviation 0.1')

        # Stacked state vector [awareness-(n_nodesx1) pos-(3x1) vel-(3x1) accel-(3x1) heading-(1x1) t_sum-(1x1)]
        self.perturbed_rewards = x0
        x0 = np.append(x0,[float(self.start_node[0]),float(self.start_node[1]) ,float(self.start_node[2]), 0,0,0, 0,0,0, 0, 0])
        
        heading_vector = SX(2, 1)
        heading_vector[0] = cos(uav_heading)
        heading_vector[1] = sin(uav_heading)
        heading_cost =  -dot(heading_vector, uav_vel[0:2])
        A_tot += 1.5 * heading_cost      # Hardcoded scaling factor 
        A_final = A_tot + 1000*(uav_pos[0]-float(self.end_node[0]))**2 + 1000*(uav_pos[1]-float(self.end_node[1]))**2 + 1000*(uav_pos[2]-float(self.end_node[2]))**2     # terminal cost

        model.set_rhs('awareness', awareness_next)
        model.set_expression('heading_cost', heading_cost)
        model.set_expression('A_tot', A_tot)
        model.set_expression('A_final', A_final)
        model.setup()
        print('Model initialized')
        self.model = model

        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': self.t_step,
            # 'n_robust': 0,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.linear_solver': self.solver,'ipopt.print_level': 5, 'ipopt.max_iter': 100000, 'ipopt.timing_statistics': 'yes'}
        }
        mpc.set_param(**setup_mpc)

        # Cost Function
        lterm = model.aux['A_tot']
        mterm = model.aux['A_final']
        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(uav_jerk=np.array([[self.r_term], [self.r_term], [self.r_term]]), uav_heading_rate=self.r_term)

        # Constraints
        bound_padding = 3 # UAV position bounds [m]
        nodes_range = range(len(self.nodes))
        self.x_lb = min([float(self.nodes[x][0])]for x in nodes_range)[0] - bound_padding
        self.x_ub = max([float(self.nodes[x][0])]for x in nodes_range)[0] + bound_padding
        self.y_lb = min([float(self.nodes[x][1])]for x in nodes_range)[0] - bound_padding
        self.y_ub = max([float(self.nodes[x][1])]for x in nodes_range)[0] + bound_padding        
        self.z_lb = min([float(self.nodes[x][2])]for x in nodes_range)[0] - bound_padding
        self.z_ub = max([float(self.nodes[x][2])]for x in nodes_range)[0] + bound_padding
        mpc.bounds['lower', '_x', 'uav_pos'] = np.array([[self.x_lb], [self.y_lb], [self.z_lb]])
        mpc.bounds['upper', '_x', 'uav_pos'] = np.array([[self.x_ub], [self.y_ub], [self.z_ub]])
        mpc.bounds['lower', '_u', 'uav_heading_rate'] = -self.max_heading_rate
        mpc.bounds['upper', '_u', 'uav_heading_rate'] = self.max_heading_rate
        mpc.bounds['lower', '_u', 'time_dynamic'] = self.dt_min 
        mpc.bounds['upper', '_u', 'time_dynamic'] = self.dt_max
        mpc.bounds['lower', '_x', 't_sum'] = 0
        mpc.bounds['upper', '_x', 't_sum'] = self.travel_budget

        # Necessary for formulation as OP
        mpc.terminal_bounds['lower', 't_sum'] = self.travel_budget
        mpc.terminal_bounds['upper', 't_sum'] = self.travel_budget

        # Terminal bounds need to be set explicitly to handle a bug in the do_mpc library
        mpc.terminal_bounds['lower', 'uav_vel'] = np.array([[-self.v_max / np.sqrt(3)], [-self.v_max / np.sqrt(3)], [-self.v_max / np.sqrt(3)]])
        mpc.terminal_bounds['upper', 'uav_vel'] = np.array([[self.v_max / np.sqrt(3)], [self.v_max / np.sqrt(3)], [self.v_max] / np.sqrt(3)])
        mpc.terminal_bounds['lower', 'uav_accel'] = np.array([[-self.a_max / np.sqrt(3)], [-self.a_max / np.sqrt(3)], [-self.a_max / np.sqrt(3)]])
        mpc.terminal_bounds['upper', 'uav_accel'] = np.array([[self.a_max / np.sqrt(3)], [self.a_max / np.sqrt(3)], [self.a_max / np.sqrt(3)]])


        # Magnitude constraints on velocity, acceleration and jerk
        uav_vel_con = mpc.set_nl_cons('uav_vel', uav_vel[0]**2 + uav_vel[1]**2 + uav_vel[2]**2, self.v_max**2)
        uav_accel_con = mpc.set_nl_cons('uav_acc', uav_accel[0]**2 + uav_accel[1]**2 + uav_accel[2]**2, self.a_max**2)
        uav_jerk_con = mpc.set_nl_cons('uav_jerk', uav_jerk[0]**2 + uav_jerk[1]**2 + uav_jerk[2]**2, self.j_max**2)

        mpc.setup()

        mpc.ub_opt_x['_u', self.n_horizon - 1, 0, 'uav_jerk'] = np.array([[0.0], [0.0], [0.0]])     # Terminal constraint for jerk

        self.x0 = np.array(x0)
        mpc.x0 = x0
        mpc.u0 = np.array([0.0,0.0,0.0, 0.0, 0.1])
        mpc.set_initial_guess()
        self.mpc = mpc

    def plan_full(self):
        try:           
            self.mpc.u0 = self.mpc.make_step(self.x0)
            self.mpc_data = self.mpc.data
            time_predicted = self.mpc.data.prediction(('_x', 't_sum')).flatten()
            print('Total time of flight {} [s]'.format(time_predicted[-1]))
            uav_pos = self.mpc.data.prediction(('_x', 'uav_pos'))
            uav_vel = self.mpc.data.prediction(('_x', 'uav_vel'))
            uav_heading = self.mpc.data.prediction(('_x', 'uav_heading')).flatten()
            uav_heading_rate = self.mpc.data.prediction(('_u', 'uav_heading_rate')).flatten()
            uav_heading_rate = np.insert(uav_heading_rate, 0, [0.0], axis=0)
            uav_pos_x = [x[0] for x in uav_pos[0]]
            uav_pos_y = [x[0] for x in uav_pos[1]]
            uav_pos_z = [x[0] for x in uav_pos[2]]

            awareness = self.mpc.data.prediction(('_x', 'awareness'))
            collected_threshold = 0.1
            aw_collected = str(sum(self.x0[:len(self.nodes)][(awareness[:,-1].flatten()) < collected_threshold]))
            print('Collected reward: ', aw_collected)

            spline_x = CubicSpline(time_predicted, uav_pos_x)
            spline_y = CubicSpline(time_predicted, uav_pos_y)
            spline_z = CubicSpline(time_predicted, uav_pos_z)
            spline_heading = CubicSpline(time_predicted, uav_heading)

            assert time_predicted[-1] > self.dt_min
            dt = np.arange(0, time_predicted[-1], self.dt_min)
            dt = np.append(dt, time_predicted[-1])                          # np.arange does not include the last value i.e. time_predicted[-1]
            uav_interp_x = spline_x(dt)
            uav_interp_y = spline_y(dt)
            uav_interp_z = spline_z(dt)
            uav_interp_heading = spline_heading(dt)

            self.trajectory_data = [
                    [uav_interp_x[i], uav_interp_y[i], uav_interp_z[i], uav_interp_heading[i], dt[i]] for i in range(len(dt))
                ]
            self.trajectory_data = np.array(self.trajectory_data).T

            if self.visualize:
                self.visualize_trajectory(mpc_data=self.mpc.data, trajectory=self.trajectory_data)
            
            return True
    
        except Exception as e:
            print('Exception when planning full trajectory.')
            traceback.print_exc()
            return False
        
    def load_params(self, param_file_name):
        with open(f'../config/{param_file_name}', 'r') as params_file:
            run_params = yaml.safe_load(params_file)
            return run_params
        
    def visualize_trajectory(self, mpc_data, trajectory):
        instance = self.run_params['instance']
        time = mpc_data.prediction(('_x', 't_sum')).flatten()
        awareness = mpc_data.prediction(('_x', 'awareness'))

        uav_pos = mpc_data.prediction(('_x', 'uav_pos'))
        uav_vel = mpc_data.prediction(('_x', 'uav_vel'))
        uav_accel = mpc_data.prediction(('_x', 'uav_accel'))
        uav_heading = mpc_data.prediction(('_x', 'uav_heading')).flatten()
        time = mpc_data.prediction(('_x', 't_sum')).flatten()
        # Append zero initial value to all inputs
        uav_jerk = mpc_data.prediction(('_u', 'uav_jerk'))
        uav_heading_rate = mpc_data.prediction(('_u', 'uav_heading_rate')).flatten()
        timesteps = mpc_data.prediction(('_u', 'time_dynamic')).flatten()
        awareness = mpc_data.prediction(('_x', 'awareness'))
        
        # np.insert(uav_jerk, 0, np.array([[0.0], [0.0], [0.0]]))
        rewards = np.array(self.rewards)
        
        gain = sum(rewards[:len(self.nodes)][(awareness[:,-1].flatten()) < 0.1])
        
        # Plot Dynamic Trajectory 3D with all parameters
        
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(3, 4, (3, 12), projection='3d')
        ax.w_xaxis.set_pane_color((0.05, 0.05, 0.05, 0.0))
        ax.w_yaxis.set_pane_color((0.05, 0.05, 0.05, 0.0))
        ax.w_zaxis.set_pane_color((0.05, 0.05, 0.05, 0.0))
        ax.grid(color='black', linestyle='-')
        uav_pos = mpc_data.prediction(('_x', 'uav_pos'))
        uav_vel = mpc_data.prediction(('_x', 'uav_vel'))
        uav_accel = mpc_data.prediction(('_x', 'uav_accel'))
        uav_heading = mpc_data.prediction(('_x', 'uav_heading')).flatten()
        time = mpc_data.prediction(('_x', 't_sum')).flatten()
        uav_jerk = mpc_data.prediction(('_u', 'uav_jerk'))
        uav_heading_rate = mpc_data.prediction(('_u', 'uav_heading_rate')).flatten()
        timesteps = mpc_data.prediction(('_u', 'time_dynamic')).flatten()
        awareness = mpc_data.prediction(('_x', 'awareness'))
        
        rewards = np.array(self.rewards)
        gain = sum(rewards[:len(self.nodes)][(awareness[:,-1].flatten()) < 0.1])
        suptitle_string = 'Instance: {} | Budget: {} [s] | Gain: {}'.format(instance, time[-1], gain)
        plt.suptitle(suptitle_string)
        plt.title('3D-KOP with Time Adaptive MPC')

        # Plot start and end nodes        
        ax.scatter3D(float(self.start_node[0]), float(self.start_node[1]), float(self.start_node[2]), color='red', s=35, alpha=1.0, label='start')
        ax.scatter3D(float(self.end_node[0]), float(self.end_node[1]), float(self.end_node[2]), color='green', s=35, alpha=1.0, label='end')
        for l in range(len(self.nodes)):
                i = float(self.nodes[l][0])
                j = float(self.nodes[l][1])
                k = float(self.nodes[l][2])
                ax.scatter3D(i, j, k, color='green', s=35)

        # Plot points of the executed trajectory
        ax.plot3D(trajectory[0], trajectory[1], trajectory[2], color='black', linewidth=2, linestyle='dashed')
        ax.scatter3D(uav_pos[0,:], uav_pos[1,:], uav_pos[2,:], color='grey', s=8, alpha=0.5, label='Sampled MP')
        ax.set_zlim3d(bottom=0.0, top=10.0)
        points = np.array([uav_pos[0,:], uav_pos[1,:], uav_pos[2,:]]).T.reshape(-1, 1, 3)
        print(points.shape)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        velocities = []
        for i in range(uav_pos.shape[1] - 1):
            dx = np.sqrt((uav_pos[0,i][0] - uav_pos[0,i+1][0])**2 + (uav_pos[1,i][0] - uav_pos[1,i+1][0])**2 + (uav_pos[2,i][0] - uav_pos[2,i+1][0])**2)
            velocities.append(dx/timesteps[i])
        velocities = np.array(velocities)
        norm = plt.Normalize(velocities.min(), velocities.max())
        cmap = cm.get_cmap('viridis')
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(velocities)
        lc.set_linewidth(2)
        line = ax.add_collection3d(lc)
        ax.set_title('3D Trajectory with Velocity Profile')
        ax.legend()

        ax = fig.add_subplot(3, 4, 1)
        net_velocity = np.sqrt(uav_vel[0,:].flatten()**2 + uav_vel[1,:].flatten()**2 + uav_vel[2,:].flatten()**2) 
        ax.plot(time, uav_vel[0,:].flatten(), label='x', color='red')
        ax.plot(time, uav_vel[1,:].flatten(), label='y', color='blue')
        ax.plot(time, uav_vel[2,:].flatten(), label='z', color='orange')
        ax.plot(time, net_velocity, label='net', color='green')
        ax.set_title('Velocity')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('vel [m/s]')
        ax.legend()

        ax = fig.add_subplot(3, 4, 2)
        net_accel = np.sqrt(uav_accel[0,:].flatten()**2 + uav_accel[1,:].flatten()**2 + uav_accel[2,:].flatten()**2) 
        ax.plot(time, uav_accel[0,:].flatten(), label='x', color='red')
        ax.plot(time, uav_accel[1,:].flatten(), label='y', color='blue')
        ax.plot(time, uav_accel[2,:].flatten(), label='z', color='orange')
        ax.plot(time, net_accel, label='net', color='green')
        ax.set_title('Acceleration')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('acc [m/s^2]')
        ax.legend()

        ax = fig.add_subplot(3, 4, 5)
        net_jerk = np.sqrt(uav_jerk[0,:].flatten()**2 + uav_jerk[1,:].flatten()**2 + uav_jerk[2,:].flatten()**2) 
        # Append zero initial value to all inputs
        ax.plot(time, np.insert(uav_jerk[0,:].flatten(), 0, 0.0), label='x', color='red')
        ax.plot(time, np.insert(uav_jerk[1,:].flatten(), 0, 0.0), label='y', color='blue')
        ax.plot(time, np.insert(uav_jerk[2,:].flatten(), 0, 0.0), label='z', color='orange')
        ax.plot(time, np.insert(net_jerk, 0, 0), label='net', color='green')
        ax.set_title('Jerk')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('jerk [m/s^3]')
        ax.legend()

        ax = fig.add_subplot(3, 4, 6)
        ax.plot(time, np.insert(timesteps, 0, 0), label='dt', color='green')
        ax.set_title('Variable Timestep')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('var dt [s]')
        ax.legend()

        ax = fig.add_subplot(3, 4, 9)
        ax.plot(time, uav_heading, label='heading', color='orange')
        ax.set_title('Heading')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('heading [rad]')
        ax.legend()

        ax = fig.add_subplot(3, 4, 10)
        ax.plot(time, np.insert(uav_heading_rate, 0, 0) , label='heading rate', color='red')
        ax.set_title('Heading Rate')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('heading rate [rad/s]')
        ax.legend()
        
        def on_resize(event):
            fig.tight_layout()
            fig.subplots_adjust(top=0.90)
            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('resize_event', on_resize)

        fig.subplots_adjust(top=0.90)
        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    try:
        planner = mpcOfflinePlanner(visualize=True)
        # Load params from yaml file with relative path ../config/
        run_params = planner.load_params(param_file_name='run_params.yaml')
        planner.plan(run_params)
    except Exception as e:
        print(f'Exception {e} occured')
