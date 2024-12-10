#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from mfboTrajectory.utils import *
# from mfboTrajectory.agents import ActiveMFDGP
from mfboTrajectory.minSnapTrajectoryPolytopes import MinSnapTrajectoryPolytopes
from mfboTrajectory.multiFidelityModelPolytopes import get_waypoints_plane, meta_low_fidelity, meta_high_fidelity, get_dataset_init, check_dataset_init, meta_low_fidelity_multi, get_dataset_init_multi
from mfboTrajectory.utilsConvexDecomp import *
from mfboTrajectory.agents_two_drones import TwoDrone

if __name__ == "__main__":
    sample_name = [['traj_13', 'traj_14']]
    drone_model = "default"
    
    rand_seed = [123, 445, 678, 115, 92, 384, 992, 874, 490, 41, 83, 78, 991, 993, 994, 995, 996, 997, 998, 999]
    max_col_err = 0.03 
    N_trial=3  # number of consecutive successful flights needed to be considered valid (used in sanity_check)
    
    parser = argparse.ArgumentParser(description='mfbo experiment')
    parser.add_argument('-l', dest='flag_load_exp_data', action='store_true', help='load exp data')
    parser.add_argument('-g', dest='flag_switch_gpu', action='store_true', help='switch gpu to gpu 1')
    parser.add_argument('-t', "--sample_idx", type=int, help="assign model index", default=0)
    parser.add_argument("-s", "--seed_idx", type=int, help="assign seed index", default=0)
    parser.add_argument("-y", "--yaw_mode", type=int, help="assign yaw mode", default=0)
    parser.add_argument("-m", "--max_iter", type=int, help="assign maximum iteration", default=50)
    parser.add_argument("-o", "--qp_optimizer", type=str, help="select optimizer for quadratic programming", default='osqp')
    parser.set_defaults(flag_load_exp_data=False)
    parser.set_defaults(flag_switch_gpu=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use gpu if available
    torch.autograd.set_detect_anomaly(True)
    
    # PARSE ARGPARSE
    qp_optimizer = args.qp_optimizer.lower()  # default: osqp
    assert qp_optimizer in ['osqp', 'gurobi','cvxopt']
    
    yaw_mode = args.yaw_mode  # default: 0
    sample_name_ = sample_name[args.sample_idx]  # default: traj_13, traj_14
    rand_seed_ = rand_seed[args.seed_idx]  # default: 123
    max_iter = np.int(args.max_iter)  # default: 50
    
    print(f"Trajectory {sample_name_}")
    polygon_filedir = './constraints_data'
    polygon_filename = 'polytopes_constraints.yaml'
    
    poly = MinSnapTrajectoryPolytopes(drone_model=drone_model, yaw_mode=yaw_mode, qp_optimizer=qp_optimizer)
    
    # LOAD POLYTOPE ENVIRONMENT
    # plane_pos_set = list of dicts of polytopes, each dict includes list of faces ("constraint planes") and input and/or output plane
    # points = waypoints generated from polytopes (??), includes midpoint of input/output planes
    # t_set_sta = initial time allocation per segment
    points1, plane_pos_set1, t_set_sta1, waypoints1 = get_waypoints_plane(polygon_filedir, polygon_filename, sample_name_[0], flag_t_set=True)
    points2, plane_pos_set2, t_set_sta2, waypoints2 = get_waypoints_plane(polygon_filedir, polygon_filename, sample_name_[1], flag_t_set=True)

    lb = 0.1
    ub = 1.9

    # if t_set_sta1.shape[0] != t_set_sta2.shape[0]:
    #     print("Time allocation mismatch")
    #     exit()
    
    # t_dim = t_set_sta1.shape[0]
    # lb_i = np.ones(t_dim)*lb
    # ub_i = np.ones(t_dim)*ub

    # # LOAD RESULTS FROM MIN-SNAP FUNCTIONS (UPDATE_TRAJ + OPTIMIZE_ALPHA)
    # with open(f"optimize_alpha_scaled_{sample_name_[0]}.npy", "rb") as f:
    #     t_set_sim1 = np.load(f)
    #     d_ordered1 = np.load(f)
    #     d_ordered_yaw1 = np.load(f)
    #     alpha_sim1 = np.load(f)
    # with open(f"optimize_alpha_scaled_{sample_name_[1]}.npy", "rb") as f:
    #     t_set_sim2 = np.load(f)
    #     d_ordered2 = np.load(f)
    #     d_ordered_yaw2 = np.load(f)
    #     alpha_sim2 = np.load(f)


    # low_fidelity_1 = lambda x, debug=True, multicore=False: \
    #     meta_low_fidelity(poly, x, t_set_sta1, points1, plane_pos_set1, waypoints1, debug, lb=lb, ub=ub, multicore=multicore)
    # low_fidelity_2 = lambda x, debug=True, multicore=False: \
    #     meta_low_fidelity(poly, x, t_set_sta2, points2, plane_pos_set2, waypoints2, debug, lb=lb, ub=ub, multicore=multicore)
    # low_fidelity_multi = lambda x1, x2, debug=True, multicore=True: \
    #     meta_low_fidelity_multi(poly, x1, t_set_sta1, points1, plane_pos_set1, waypoints1, x2, t_set_sta2, points2, plane_pos_set2, waypoints2, debug, lb=lb, ub=ub, multicore=multicore)
    
    # # LOAD RESULTS FROM GET_DATASET_INIT (FOR DRONE 1, DRONE 2, AND DRONE 1+2)
    # with open("traj_13_init_dataset.npy", "rb") as f:
    #     X1 = np.load(f)
    #     Y1 = np.load(f)
    # with open("traj_14_init_dataset.npy", "rb") as f:
    #     X2 = np.load(f)
    #     Y2 = np.load(f)
    # with open("two_drone_init_dataset.npy", "rb") as f:
    #     X12 = np.load(f)
    #     Y12 = np.load(f)
    # # X12 = X12.reshape(-1, 2*X12.shape[1])

    # np.random.seed(rand_seed_)
    # torch.manual_seed(rand_seed_)

    # # filenames
    # fileprefix = "test_polytopes"
    # filedir = f"./mfbo_data/{sample_name_}"
    # logprefix = '{sample_name_}/{fileprefix}/{rand_seed_}'
    # results_filename = f'result_{fileprefix}_{rand_seed_}.yaml'
    # exp_data_filename = f'exp_data_{fileprefix}_{rand_seed_}.yaml'


    # # flag_check = check_result_data(filedir, results_filename, max_iter)
    #     # create agent
    # two_drones = TwoDrone(
    #     X1 = X1,
    #     Y1 = Y1,
    #     X2 = X2,
    #     Y2 = Y2,
    #     X12 = X12,
    #     Y12 = Y12,
    #     lb_i = lb_i,
    #     ub_i = ub_i,
    #     beta = 3.0,
    #     rand_seed = rand_seed_,
    #     N_cand = 128,
    #     batch_size = 1024,
    #     model_prefix = logprefix,
    #     t_set_sim_1 = t_set_sim1,
    #     t_set_sim_2 = t_set_sim2,
    #     eval_func_1 = low_fidelity_1,
    #     eval_func_2 = low_fidelity_2,
    #     eval_func_12 = low_fidelity_multi
    # )
    # X, Y = two_drones.bayes_opt()
    X = np.array([[0.62682851,0.57339644,0.66341361,0.40905405,0.44583309,0.75439185,0.48987495,0.58259271]])
    Y = 1
    alpha_set_1 = lb + X.flatten()[range(4)] * (ub - lb)
    print(alpha_set_1)
    alpha_set_2 = lb + X.flatten()[range(4, 8)] * (ub - lb)
    t_set_1, d_ordered_1, d_ordered_yaw_1 = poly.update_traj(t_set_sta1, 
                                                                points1, 
                                                                plane_pos_set1, 
                                                                waypoints1,
                                                                alpha_set=alpha_set_1,
                                                                flag_fixed_point=False, 
                                                                flag_fixed_end_point=False)
    t_set_2, d_ordered_2, d_ordered_yaw_2 = poly.update_traj(t_set_sta2, 
                                                                points2, 
                                                                plane_pos_set2, 
                                                                waypoints2,
                                                                alpha_set=alpha_set_2,
                                                                flag_fixed_point=False, 
                                                                flag_fixed_end_point=False)

    with open("final_traj_13_run2.npy", "wb") as f:
        np.save(f, t_set_1)
        np.save(f, d_ordered_1)
        np.save(f, d_ordered_yaw_1)
    with open("final_traj_14_run2.npy", "wb") as f:
        np.save(f, t_set_2)
        np.save(f, d_ordered_2)
        np.save(f, d_ordered_yaw_2)
    with open("final_traj_13_run2.npy", "rb") as f:
        t = np.load(f)
        d = np.load(f)
        yaw = np.load(f)
    with open("final_traj_14_run2.npy", "rb") as f:
        t = np.load(f)
        d = np.load(f)
        yaw = np.load(f)

    print(t)
    
