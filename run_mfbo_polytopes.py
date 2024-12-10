#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from mfboTrajectory.utils import *
from mfboTrajectory.agents import ActiveMFDGP
from mfboTrajectory.minSnapTrajectoryPolytopes import MinSnapTrajectoryPolytopes
from mfboTrajectory.multiFidelityModelPolytopes import get_waypoints_plane, meta_low_fidelity, meta_high_fidelity, get_dataset_init, check_dataset_init
from mfboTrajectory.utilsConvexDecomp import *

if __name__ == "__main__":
    sample_name = ['traj_9', 'traj_10', 'traj_11', 'traj_12', 'traj_13', 'traj_14']
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

    # if args.flag_switch_gpu:
    #     torch.cuda.set_device(1)
    # else:
    #     torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use gpu if available
    torch.autograd.set_detect_anomaly(True)
    
    # PARSE ARGPARSE
    # select optimizer
    qp_optimizer = args.qp_optimizer.lower()
    assert qp_optimizer in ['osqp', 'gurobi','cvxopt']
    
    yaw_mode = args.yaw_mode
    sample_name_ = sample_name[args.sample_idx]
    rand_seed_ = rand_seed[args.seed_idx]
    max_iter = np.int(args.max_iter)
    print(f"MAX_ITER: {max_iter}")
    
    polygon_filedir = './constraints_data'
    polygon_filename = 'polytopes_constraints.yaml'
    
    poly = MinSnapTrajectoryPolytopes(drone_model=drone_model, yaw_mode=yaw_mode, qp_optimizer=qp_optimizer)
    
    # get info about polytope constraints
    # plane_pos_set = list of dicts of polytopes, each dict includes list of faces ("constraint planes") and input and/or output plane
    # points = waypoints generated from polytopes (??), includes midpoint of input/output planes
    # t_set_sta = initial time allocation per segment
    points_13, plane_pos_set, t_set_sta_13, waypoints_13 = get_waypoints_plane(polygon_filedir, polygon_filename, "traj_13", flag_t_set=True)
    points_14, plane_pos_set, t_set_sta_14, waypoints_14 = get_waypoints_plane(polygon_filedir, polygon_filename, "traj_14", flag_t_set=True)

    lb = 0.1
    ub = 1.9
    # if yaw_mode == 0:
    #     flag_yaw_zero = True
    # else:
    #     flag_yaw_zero = False
    
    if yaw_mode == 0:
        sample_name_ += "_yaw_zero"
    print("Yaw_mode {}".format(yaw_mode))
    
    t_dim = t_set_sta_13.shape[0]
    lb_i = np.ones(t_dim)*lb
    ub_i = np.ones(t_dim)*ub

    # # look for existing dataset
    # res_init, data_init = check_dataset_init(sample_name_, t_dim, N_L=1000, N_H=20, lb=lb, ub=ub, sampling_mode=2)
    # print(res_init)  # false
    # print(data_init)  # None
    # res_init = False
    
    # if res_init:
    #     alpha_sim, X_L, Y_L, X_H, Y_H = data_init
    #     t_set_sim = t_set_sta * alpha_sim
    #     low_fidelity = lambda x, debug=True, multicore=False: \
    #         meta_low_fidelity(poly, x, t_set_sta, points, plane_pos_set, debug, lb=lb, ub=ub, multicore=multicore)
    #     high_fidelity = lambda x, return_snap=False, multicore=False: \
    #         meta_high_fidelity(poly, x, t_set_sim, points, plane_pos_set, lb=lb, ub=ub, \
    #             return_snap=return_snap, multicore=multicore, \
    #             max_col_err=max_col_err, N_trial=N_trial)
    print("Initializing dataset")
    # sanity_check_t = lambda t_set, d_ordered, d_ordered_yaw: \
    #     poly.run_sim_loop(t_set, d_ordered, d_ordered_yaw, plane_pos_set, max_col_err=max_col_err, N_trial=N_trial)
    sanity_check_t = None
    print("Start generating initial trajectory")
    t_set_sta_13, d_ordered_13, d_ordered_yaw_13 = poly.update_traj(t_set_sta_13, 
                                                            points_13, 
                                                            plane_pos_set, 
                                                            waypoints_13,
                                                            np.ones_like(t_set_sta_13),
                                                            flag_fixed_point=False, 
                                                            flag_fixed_end_point=False)
    
    t_set_sta_14, d_ordered_14, d_ordered_yaw_14 = poly.update_traj(t_set_sta_14, 
                                                            points_14, 
                                                            plane_pos_set, 
                                                            waypoints_14,
                                                            np.ones_like(t_set_sta_14),
                                                            flag_fixed_point=False, 
                                                            flag_fixed_end_point=False)
    
    with open(f"update_traj_13.npy", "wb") as f:
        np.save(f, t_set_sta_13)
        np.save(f, d_ordered_13)
        np.save(f, d_ordered_yaw_13)
    
    with open(f"update_traj_14.npy", "wb") as f:
        np.save(f, t_set_sta_14)
        np.save(f, d_ordered_14)
        np.save(f, d_ordered_yaw_14)

    # ensure time allocations match (eq 28)
    # note: for 13 and 14, time allocations always match (both are [5, 10, 10, 5])
    if not np.array_equal(t_set_sta_13, t_set_sta_14):
        print("scaling time arrays")
        waypoints_idx = np.where(waypoints_13)[0]  # waypoints are the same between both files

        for idx in range(len(waypoints_idx)-1):
            sum_xij_13 = sum(t_set_sta_13[waypoints_idx[idx]:waypoints_idx[idx+1]])
            sum_xij_14 = sum(t_set_sta_14[waypoints_idx[idx]:waypoints_idx[idx+1]])

            scaled_13 = np.array(t_set_sta_13[waypoints_idx[idx]:waypoints_idx[idx+1]]) * (2*sum_xij_14) / (sum_xij_13 + sum_xij_14)
            scaled_14 = np.array(t_set_sta_14[waypoints_idx[idx]:waypoints_idx[idx+1]]) * (2*sum_xij_13) / (sum_xij_13 + sum_xij_14)

            t_set_sta_13[waypoints_idx[idx]:waypoints_idx[idx+1]] = list(scaled_13)
            t_set_sta_14[waypoints_idx[idx]:waypoints_idx[idx+1]] = list(scaled_14)


    print("Done generating initial trajectory")
    optimize_alpha = True # runs optimize_alpha if true, reads from file if false
    if optimize_alpha:
        print("Start generating time optimized trajectory")
        t_set_sim_13, d_ordered_13, d_ordered_yaw_13, alpha_sim_13 = poly.optimize_alpha(points_13,  
                                                                                t_set_sta_13,  # initial time array
                                                                                d_ordered_13,  # initial position array
                                                                                d_ordered_yaw_13,  # initial yaw angle array
                                                                                alpha_scale=1.0,  # alpha scaling
                                                                                sanity_check_t=sanity_check_t,  # sim env to check valid trajectory
                                                                                flag_return_alpha=True,  # returns alpha value if true
                                                                                )
        # save data to files
        with open(f"optimize_alpha_traj_13.npy", "wb") as f:
            np.save(f, t_set_sim_13)
            np.save(f, d_ordered_13)
            np.save(f, d_ordered_yaw_13)
            np.save(f, np.array(alpha_sim_13))

        print("alpha_sim 13: {}".format(alpha_sim_13))

        t_set_sim_14, d_ordered_14, d_ordered_yaw_14, alpha_sim_14 = poly.optimize_alpha(points_14,  
                                                                                t_set_sta_14,  # initial time array
                                                                                d_ordered_14,  # initial position array
                                                                                d_ordered_yaw_14,  # initial yaw angle array
                                                                                alpha_scale=1.0,  # alpha scaling
                                                                                sanity_check_t=sanity_check_t,  # sim env to check valid trajectory
                                                                                flag_return_alpha=True,  # returns alpha value if true
                                                                                )
        # save data to files
        with open(f"optimize_alpha_traj_14.npy", "wb") as f:
            np.save(f, t_set_sim_14)
            np.save(f, d_ordered_14)
            np.save(f, d_ordered_yaw_14)
            np.save(f, np.array(alpha_sim_14))

        print("alpha_sim 14: {}".format(alpha_sim_14))
    else:
        with open(f"optimize_alpha_traj_13.npy", "rb") as f:
            t_set_sim_13 = np.load(f)
            d_ordered_13_sim = np.load(f)
            d_ordered_yaw_13_sim = np.load(f)
            alpha_sim_13 = np.load(f)
        with open(f"optimize_alpha_traj_14.npy", "rb") as f:
            t_set_sim_14 = np.load(f)
            d_ordered_14_sim = np.load(f)
            d_ordered_yaw_14_sim = np.load(f)
            alpha_sim_14 = np.load(f)

    # ENFORCE EQUAL SCALING
    if alpha_sim_13 != alpha_sim_14:
        print("scaling alphas")
        alpha = max(alpha_sim_13, alpha_sim_14)
        N_wp = np.int(d_ordered_13.shape[0]/poly.N_DER) # num waypoints

        t_set_sta_13 = t_set_sta_13 * alpha  # times scaled by alpha (??)
        t_set_sta_14 = t_set_sta_14 * alpha  # times scaled by alpha (??)
        
        # scale d ordered by alpha
        d_ordered_13 = poly.get_alpha_matrix(alpha, N_wp).dot(d_ordered_13)  # list of positions
        if np.all(d_ordered_yaw_13 != None):  # list of yaw angles
            d_ordered_yaw_13 = poly.get_alpha_matrix_yaw(alpha,N_wp).dot(d_ordered_yaw_13)
        else:
            d_ordered_yaw_13 = None

        d_ordered_14 = poly.get_alpha_matrix(alpha, N_wp).dot(d_ordered_14)  # list of positions
        if np.all(d_ordered_yaw_14 != None):  # list of yaw angles
            d_ordered_yaw_14 = poly.get_alpha_matrix_yaw(alpha,N_wp).dot(d_ordered_yaw_14)
        else:
            d_ordered_yaw_14 = None
        
    with open(f"optimize_alpha_scaled_traj_13.npy", "wb") as f:
        np.save(f, t_set_sta_13)
        np.save(f, d_ordered_13)
        np.save(f, d_ordered_yaw_13)
        np.save(f, np.array(alpha))

    print("alpha_sim 13: {}".format(alpha))

    with open(f"optimize_alpha_scaled_traj_14.npy", "wb") as f:
        np.save(f, t_set_sta_14)
        np.save(f, d_ordered_14)
        np.save(f, d_ordered_yaw_14)
        np.save(f, np.array(alpha_sim_14))

    print("alpha_sim 14: {}".format(alpha))