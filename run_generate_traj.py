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

    # X = np.array([[0.62682851,0.57339644,0.66341361,0.40905405,0.44583309,0.75439185,0.48987495,0.58259271]])  # from traj_13_init_dataset etc
    X = np.array([0.65545811,0.40646523,0.8802627,0.85447426,0.58979782,0.47212551,0.55526965,1.17946731])  # from traj_13_optimize_alpha_scaled_init_dataset etc
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
    filenames = [f"final_{sn}_run3.npy" for sn in sample_name[0]]
    with open(filenames[0], "wb") as f:
        np.save(f, t_set_1)
        np.save(f, d_ordered_1)
        np.save(f, d_ordered_yaw_1)
    with open(filenames[1], "wb") as f:
        np.save(f, t_set_2)
        np.save(f, d_ordered_2)
        np.save(f, d_ordered_yaw_2)

    # ensure files were created properly by reading in
    with open(filenames[0], "rb") as f:
        t = np.load(f)
        d = np.load(f)
        yaw = np.load(f)
    print(f"saved {filenames[0]}")

    with open(filenames[1], "rb") as f:
        t = np.load(f)
        d = np.load(f)
        yaw = np.load(f)
    print(f"saved {filenames[1]}")
