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
from mfboTrajectory.agents import ActiveMFDGP, active_learning
from mfboTrajectory.minSnapTrajectoryPolytopes import MinSnapTrajectoryPolytopes
from mfboTrajectory.multiFidelityModelPolytopes import get_waypoints_plane, meta_low_fidelity, meta_high_fidelity, get_dataset_init, check_dataset_init, meta_low_fidelity_multi, get_dataset_init_multi
from mfboTrajectory.utilsConvexDecomp import *

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
    
    print(f"Trajectory {sample_name_}")
    polygon_filedir = './constraints_data'
    polygon_filename = 'polytopes_constraints.yaml'
    
    poly = MinSnapTrajectoryPolytopes(drone_model=drone_model, yaw_mode=yaw_mode, qp_optimizer=qp_optimizer)
    
    # get info about polytope constraints
    # plane_pos_set = list of dicts of polytopes, each dict includes list of faces ("constraint planes") and input and/or output plane
    # points = waypoints generated from polytopes (??), includes midpoint of input/output planes
    # t_set_sta = initial time allocation per segment
    points1, plane_pos_set1, t_set_sta1, waypoints1 = get_waypoints_plane(polygon_filedir, polygon_filename, sample_name_[0], flag_t_set=True)
    points2, plane_pos_set2, t_set_sta2, waypoints2 = get_waypoints_plane(polygon_filedir, polygon_filename, sample_name_[1], flag_t_set=True)
    
    lb = 0.1
    ub = 1.9
    # if yaw_mode == 0:
    #     flag_yaw_zero = True
    # else:
    #     flag_yaw_zero = False
    
    # if yaw_mode == 0:
    #     sample_name_[0] += "_yaw_zero"
    #     sample_name_[1] += "_yaw_zero"
    print("Yaw_mode {}".format(yaw_mode))
    if t_set_sta1.shape[0] != t_set_sta2.shape[0]:
        print("Time allocation mismatch")
        exit()
    
    t_dim = t_set_sta1.shape[0]
    lb_i = np.ones(t_dim)*lb
    ub_i = np.ones(t_dim)*ub

    
    print("Initializing dataset")
    sanity_check_t = lambda t_set, d_ordered, d_ordered_yaw: \
        poly.run_sim_loop(t_set, d_ordered, d_ordered_yaw, plane_pos_set, max_col_err=max_col_err, N_trial=N_trial)
    print("Start generating initial trajectory")
    # t_set_sta, d_ordered, d_ordered_yaw = poly.update_traj(t_set_sta, 
    #                                                        points, 
    #                                                        plane_pos_set, 
    #                                                        waypoints,
    #                                                        np.ones_like(t_set_sta),
    #                                                        flag_fixed_point=False, 
    #                                                        flag_fixed_end_point=False)

    # print("Done generating initial trajectory")
    # print("Start generating time optimized trajectory")
    # t_set_sim, d_ordered, d_ordered_yaw, alpha_sim = poly.optimize_alpha(points,  
    #                                                                      t_set_sta,  # initial time array
    #                                                                      d_ordered,  # initial position array
    #                                                                      d_ordered_yaw,  # initial yaw angle array
    #                                                                      alpha_scale=1.0,  # alpha scaling
    #                                                                      sanity_check_t=sanity_check_t,  # sim env to check valid trajectory
    #                                                                      flag_return_alpha=True,  # returns alpha value if true
    #                                                                      )
    # save data to files
    # with open(f"{sample_name_}.npy", "wb") as f:
    #     np.save(f, t_set_sim)
    #     np.save(f, d_ordered)
    #     np.save(f, d_ordered_yaw)
    #     np.save(f, np.array(alpha_sim))
    with open(f"optimize_alpha_{sample_name_[0]}.npy", "rb") as f:
        t_set_sim1 = np.load(f)
        d_ordered1 = np.load(f)
        d_ordered_yaw1 = np.load(f)
        alpha_sim1 = np.load(f)
    
    with open(f"optimize_alpha_{sample_name_[1]}.npy", "rb") as f:
        t_set_sim2 = np.load(f)
        d_ordered2 = np.load(f)
        d_ordered_yaw2 = np.load(f)
        alpha_sim2 = np.load(f)

    # print("alpha_sim: {}".format(alpha_sim))
    
        

    low_fidelity = lambda x, debug=True, multicore=False: \
        meta_low_fidelity(poly, x, t_set_sta, points, plane_pos_set, waypoints, debug, lb=lb, ub=ub, multicore=multicore)
    low_fidelity_multi = lambda x1, x2, debug=True, multicore=True: \
        meta_low_fidelity_multi(poly, x1, t_set_sta1, points1, plane_pos_set1, waypoints1, x2, t_set_sta2, points2, plane_pos_set2, waypoints2, debug, lb=lb, ub=ub, multicore=multicore)
    print("Start initializing dataset")
    # X_L1, Y_L1, X_H, Y_H = get_dataset_init(sample_name_[0], 
    #                                         alpha_sim1, 
    #                                         low_fidelity,
    #                                         t_dim, 
    #                                         N_L=100, 
    #                                         N_H=2, 
    #                                         lb=lb, 
    #                                         ub=ub,
    #                                         sampling_mode=2, 
    #                                         flag_multicore=True)
    # X_L2, Y_L2, X_H, Y_H = get_dataset_init(sample_name_[1], 
    #                                         alpha_sim2, 
    #                                         low_fidelity,
    #                                         t_dim, 
    #                                         N_L=100, 
    #                                         N_H=2, 
    #                                         lb=lb, 
    #                                         ub=ub,
    #                                         sampling_mode=2, 
    #                                         flag_multicore=True)
    X_L_all, Y_L_all = get_dataset_init_multi(sample_name_, 
                                    alpha_sim1,
                                    alpha_sim2, 
                                    low_fidelity_multi,
                                    t_dim, 
                                    N_L=1000,
                                    lb=lb, 
                                    ub=ub,
                                    sampling_mode=2, 
                                    flag_multicore=True)
    # with open("traj_13_init_dataset.npy", "wb") as f:
    #     np.save(f, X_L)
    #     np.save(f, Y_L)
    with open("All_init_dataset.npy", "wb") as f:
        np.save(f, X_L_all)
        np.save(f, Y_L_all)
        
    # with open("traj_13_init_dataset.npy", "rb") as f:
    #     X_L = np.load(f)
    #     Y_L = np.load(f)
    # with open("traj_13_init_dataset.npy", "wb") as f:
    #     np.save(f, X_L)
    #     np.save(f, Y_L)
    np.random.seed(rand_seed_)
    torch.manual_seed(rand_seed_)

    # filenames
    fileprefix = "test_polytopes"
    filedir = f"./mfbo_data/{sample_name_}"
    logprefix = '{sample_name_}/{fileprefix}/{rand_seed_}'
    results_filename = f'result_{fileprefix}_{rand_seed_}.yaml'
    exp_data_filename = f'exp_data_{fileprefix}_{rand_seed_}.yaml'


    flag_check = check_result_data(filedir, results_filename, max_iter)
    if not flag_check:
        # create agent
        drone1 = ActiveMFDGP(
            X_L=X_L, 
            Y_L=Y_L, 
            X_H=X_H, 
            Y_H=Y_H,
            lb_i=lb_i, 
            ub_i=ub_i, 
            rand_seed=rand_seed_,
            C_L=1.0, 
            C_H=10.0,
            delta_L=0.9, 
            delta_H=0.6, 
            beta=3.0, 
            N_cand=16384,
            gpu_batch_size=1024,
            sampling_func_L=low_fidelity,
            sampling_func_H=high_fidelity,
            t_set_sim=t_set_sim,
            utility_mode=0, 
            sampling_mode=5,
            model_prefix=logprefix,
            iter_create_model=200
        )

        path_exp_data = os.path.join(filedir, exp_data_filename)
        if args.flag_load_exp_data and os.path.exists(path_exp_data):
            drone1.load_exp_data(filedir=filedir, filename=exp_data_filename)

    # Drone 2

        
    points, plane_pos_set, t_set_sta, waypoints = get_waypoints_plane(polygon_filedir, polygon_filename, "traj_14", flag_t_set=True)

    lb = 0.1
    ub = 1.9
    # if yaw_mode == 0:
    #     flag_yaw_zero = True
    # else:
    #     flag_yaw_zero = False

    if yaw_mode == 0:
        sample_name_ += "_yaw_zero"
    print("Yaw_mode {}".format(yaw_mode))

    t_dim = t_set_sta.shape[0]
    lb_i = np.ones(t_dim)*lb
    ub_i = np.ones(t_dim)*ub
    with open("traj_14_yaw_zero.npy", "rb") as f:
        t_set_sim = np.load(f)
        d_ordered = np.load(f)
        d_ordered_yaw = np.load(f)
        alpha_sim = np.load(f)

    # print("alpha_sim: {}".format(alpha_sim))

    low_fidelity = lambda x, debug=True, multicore=False: \
        meta_low_fidelity(poly, x, t_set_sta, points, plane_pos_set, waypoints, debug, lb=lb, ub=ub, multicore=multicore)
    X_L, Y_L, X_H, Y_H = get_dataset_init(sample_name_, 
                                            alpha_sim, 
                                            waypoints,
                                            low_fidelity, 
                                            t_dim, 
                                            N_L=1000, 
                                            N_H=20, 
                                            lb=lb, 
                                            ub=ub,
                                            sampling_mode=2, 
                                            flag_multicore=True)
    with open("traj_14_init_dataset.npy", "wb") as f:
        np.save(f, X_L)
        np.save(f, Y_L)
        

    # filenames
    fileprefix = "test_polytopes"
    filedir = f"./mfbo_data/{sample_name_}"
    logprefix = '{sample_name_}/{fileprefix}/{rand_seed_}'
    results_filename = f'result_{fileprefix}_{rand_seed_}.yaml'
    exp_data_filename = f'exp_data_{fileprefix}_{rand_seed_}.yaml'


    flag_check = check_result_data(filedir, results_filename, max_iter)
    if not flag_check:
        # create agent
        drone2 = ActiveMFDGP(
            X_L=X_L, 
            Y_L=Y_L, 
            X_H=X_H, 
            Y_H=Y_H,
            lb_i=lb_i, 
            ub_i=ub_i, 
            rand_seed=rand_seed_,
            C_L=1.0, 
            C_H=10.0,
            delta_L=0.9, 
            delta_H=0.6, 
            beta=3.0, 
            N_cand=16384,
            gpu_batch_size=1024,
            sampling_func_L=low_fidelity,
            sampling_func_H=high_fidelity,
            t_set_sim=t_set_sim,
            utility_mode=0, 
            sampling_mode=5,
            model_prefix=logprefix,
            iter_create_model=200
        )

        path_exp_data = os.path.join(filedir, exp_data_filename)
        if args.flag_load_exp_data and os.path.exists(path_exp_data):
            drone2.load_exp_data(filedir=filedir, filename=exp_data_filename)
        print("Start active learning")
        active_learning(drone1, drone2,
            N=max_iter, 
            plot=False, 
            MAX_low_fidelity=0,
            filedir=filedir,
            filename_result=results_filename,
            filename_exp=exp_data_filename
        )
