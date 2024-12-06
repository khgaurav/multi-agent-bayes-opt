#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, copy, time
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tensorboardX import SummaryWriter
from pyDOE import lhs

import torch
from torch.utils.data import TensorDataset, DataLoader
# from torch.nn import Linear
# import torch.nn as nn
# import torch.nn.functional as F

import gpytorch
# from gpytorch.means import ConstantMean
# from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
# from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
# from gpytorch.distributions import MultivariateNormal
# from gpytorch.models import AbstractVariationalGP, GP
from gpytorch.mlls import VariationalELBO#, AddedLossTerm
# from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
# from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from .models import *
from .trajSampler import TrajSampler, gaussian_sampler

class MFBOAgentBase():
    def __init__(self, *args, **kwargs):
        """
        Initialize the agent with given parameters.
        Parameters:
        - X_L (np.ndarray): Low-fidelity input data.
        - Y_L (np.ndarray): Low-fidelity output data.
        - X_H (np.ndarray): High-fidelity input data.
        - Y_H (np.ndarray): High-fidelity output data.
        - lb_i (np.ndarray): Lower bounds for input space.
        - ub_i (np.ndarray): Upper bounds for input space.
        - rand_seed (int): Random seed for reproducibility.
        - C_L (np.ndarray): Low-fidelity cost data.
        - C_H (np.ndarray): High-fidelity cost data.
        - sampling_func_L (callable): Sampling function for low-fidelity data.
        - sampling_func_H (callable): Sampling function for high-fidelity data.
        - t_set_sim (np.ndarray): Time set for simulation.
        - traj_wp_sampler_mean (float): Mean for trajectory waypoint sampler.
        - traj_wp_sampler_var (float): Variance for trajectory waypoint sampler.
        - delta_L (float): Delta parameter for low-fidelity data.
        - delta_H (float): Delta parameter for high-fidelity data.
        - beta (float): Beta parameter.
        - iter_create_model (int): Number of iterations to create model.
        - N_cand (int): Number of candidate samples.
        - utility_mode (int): Mode for utility function.
        - sampling_mode (int): Mode for sampling function.
        - model_prefix (str): Prefix for model name.
        - writer (SummaryWriter): TensorBoard summary writer.
        Raises:
        - Exception: If an unsupported sampling mode is provided.
        """
        # 
        self.X_L = kwargs.get('X_L', None)
        self.Y_L = kwargs.get('Y_L', None)
        self.N_L = self.X_L.shape[0]
        # self.X_H = kwargs.get('X_H', None)
        # self.Y_H = kwargs.get('Y_H', None)
        # self.N_H = self.X_H.shape[0]
        # self.N_H_i = self.X_H.shape[0]
        self.lb_i = kwargs.get('lb_i', None)
        self.ub_i = kwargs.get('ub_i', None)
        self.rand_seed = kwargs.get('rand_seed', None)
        self.C_L = kwargs.get('C_L', None)
        self.C_H = kwargs.get('C_H', None)
        self.sampling_func_L = kwargs.get('sampling_func_L', None)
        # self.sampling_func_H = kwargs.get('sampling_func_H', None)
        self.t_set_sim = kwargs.get('t_set_sim', None)
        self.traj_wp_sampler_mean = kwargs.get('traj_wp_sampler_mean', 0.5)
        self.traj_wp_sampler_var = kwargs.get('traj_wp_sampler_var', 0.2)
        
        self.delta_L = kwargs.get('delta_L', 0.8)
        # self.delta_H = kwargs.get('delta_H', 0.4)
        self.beta = kwargs.get('beta', 0.05)
        # self.dim = self.X_H.shape[1]
        self.iter_create_model = kwargs.get('iter_create_model', 200)
        self.N_cand = kwargs.get('N_cand', 1000)

        self.utility_mode = kwargs.get('utility_mode', 0)
        self.sampling_mode = kwargs.get('sampling_mode', 0)
        self.model_prefix = kwargs.get('model_prefix', 'mfbo_test')
        self.writer = SummaryWriter('runs/mfbo/'+self.model_prefix)
        
        np.random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        
        self.t_dim = self.t_set_sim.shape[0]
        self.p_dim = 0
        
        if self.sampling_mode == 0:
            self.sample_data = lambda N_sample: lhs(self.t_dim, N_sample)    
        elif self.sampling_mode == 1:
            self.traj_sampler = TrajSampler(N=self.t_dim, sigma=50.0)
            self.sample_data = lambda N_sample: self.traj_sampler.rsample(N_sample=N_sample)
        elif self.sampling_mode == 2:
            self.traj_sampler = TrajSampler(N_sample=N_cand, N=self.t_dim, sigma=50.0, flag_load=True)
            self.sample_data = lambda N_sample: self.traj_sampler.rsample(N_sample=N_sample)
        elif self.sampling_mode == 3:
            self.traj_sampler = TrajSampler(N_sample=N_cand, N=self.t_dim, sigma=1.0, flag_load=True, cov_mode=1, flag_pytorch=False)
            self.sample_data = lambda N_sample: self.traj_sampler.rsample(N_sample=N_sample)
        elif self.sampling_mode == 4:
            self.traj_sampler = TrajSampler(N_sample=N_cand, N=self.t_dim, sigma=0.5, flag_load=True, cov_mode=1, flag_pytorch=False)
            self.sample_data = lambda N_sample: self.traj_sampler.rsample(N_sample=N_sample)
        elif self.sampling_mode == 5:
            self.traj_sampler = TrajSampler(N_sample=4096, N=self.t_dim, sigma=0.2, flag_load=True, cov_mode=1, flag_pytorch=False)
            self.sample_data = lambda N_sample: self.traj_sampler.rsample(N_sample=N_sample)
        elif self.sampling_mode == 6:
            self.traj_sampler = TrajSampler(N_sample=N_cand, N=self.t_dim, sigma=20.0, flag_load=True, cov_mode=1, flag_pytorch=False)
            self.sample_data = lambda N_sample: self.traj_sampler.rsample(N_sample=N_sample)
        elif self.sampling_mode == 7:
            self.sample_data = lambda N_sample: lhs(self.t_dim, N_sample)
        else:
            raise "Not implemented"
        self.X_cand = self.sample_data(self.N_cand) # TODO FIGURE OUT THE EXACT CONTENTS OF X_CAND BECASUE ITS INDEXED LATER WITH HIGH AND LOW FIDELTIY INDICES
        
        self.min_time = 1.0
        self.min_time_cand = 1.0
        self.alpha_min = np.ones(self.X_cand.shape[1])
        self.alpha_min_cand = np.ones(self.X_cand.shape[1])
        self.flag_found_ei = False
        
        self.X_test = np.zeros((2500,2))
        xx, yy = np.meshgrid(np.linspace(0,1,50,endpoint=True),np.linspace(0,1,50,endpoint=True))
        self.X_test[:,0] = xx.reshape(-1)
        self.X_test[:,1] = yy.reshape(-1)
        
    def load_exp_data(self, \
          filedir='./mfbo_data/', \
          filename='exp_data.yaml'):
        
        yamlFile = os.path.join(filedir, filename)
        with open(yamlFile, "r") as input_stream:
            yaml_in = yaml.load(input_stream)
            self.start_iter = np.int(yaml_in["start_iter"])
            self.X_L = np.array(yaml_in["X_L"])
            self.Y_L = np.array(yaml_in["Y_L"])
            self.N_L = self.X_L.shape[0]
            # self.X_H = np.array(yaml_in["X_H"])
            # self.Y_H = np.array(yaml_in["Y_H"])
            # self.N_H = self.X_H.shape[0]
            self.X_cand = np.array(yaml_in["X_cand"])
#             self.X_cand_H = np.array(yaml_in["X_cand_H"])
            self.min_time_array = yaml_in["min_time_array"]
            self.alpha_cand_array = yaml_in["alpha_cand_array"]
            self.fidelity_array = yaml_in["fidelity_array"]
            self.found_ei_array = yaml_in["found_ei_array"]
            self.exp_result_array = yaml_in["exp_result_array"]
            self.rel_snap_array = yaml_in["rel_snap_array"]
            self.alpha_min = np.array(yaml_in["alpha_min"])
            self.min_time = np.float(self.min_time_array[-1])
            self.N_low_fidelity = np.int(yaml_in["N_low_fidelity"])
            
            prGreen("#################################################")
            prGreen("Exp data loaded. start_iter: {}, N_L: {}"\
                    .format(self.start_iter, self.Y_L.shape[0]))
            # prGreen("Exp data loaded. start_iter: {}, N_L: {}, N_H: {}"\
            #         .format(self.start_iter, self.Y_L.shape[0], self.Y_H.shape[0]))
            prGreen("#################################################")
    
    def save_exp_data(self, \
                  filedir='./mfbo_data/', \
                  filename='exp_data.yaml'):
                
        yamlFile = os.path.join(filedir, filename)
        yaml_out = open(yamlFile,"w")
        yaml_out.write("start_iter: {}\n\n".format(self.start_iter))
        
        yaml_out.write("X_L:\n")
        for it in range(self.X_L.shape[0]):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_L[it,:]])))
        yaml_out.write("\n")
        yaml_out.write("Y_L: [{}]\n".format(', '.join([str(x) for x in self.Y_L])))
        yaml_out.write("\n")
        
        # yaml_out.write("X_H:\n")
        # for it in range(self.X_H.shape[0]):
        #     yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_H[it,:]])))
        # yaml_out.write("\n")
        # yaml_out.write("Y_H: [{}]\n".format(', '.join([str(x) for x in self.Y_H])))
        # yaml_out.write("\n")
        
        yaml_out.write("X_cand:\n")
        for it in range(self.X_cand.shape[0]):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_cand[it,:]])))
        yaml_out.write("\n")
#         yaml_out.write("X_cand_H:\n")
#         for it in range(self.X_cand_H.shape[0]):
#             yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.X_cand_H[it,:]])))
#         yaml_out.write("\n")
        
        yaml_out.write("min_time_array: [{}]\n".format(', '.join([str(x) for x in self.min_time_array])))
        yaml_out.write("\n")
        yaml_out.write("alpha_cand_array:\n")
        for it in range(len(self.alpha_cand_array)):
            yaml_out.write("  - [{}]\n".format(', '.join([str(x) for x in self.alpha_cand_array[it]])))
        yaml_out.write("\n")
        yaml_out.write("fidelity_array: [{}]\n".format(', '.join([str(x) for x in self.fidelity_array])))
        yaml_out.write("\n")
        yaml_out.write("found_ei_array: [{}]\n".format(', '.join([str(x) for x in self.found_ei_array])))
        yaml_out.write("\n")
        yaml_out.write("exp_result_array: [{}]\n".format(', '.join([str(x) for x in self.exp_result_array])))
        yaml_out.write("\n")
        yaml_out.write("rel_snap_array: [{}]\n".format(', '.join([str(x) for x in self.rel_snap_array])))
        yaml_out.write("\n")
        yaml_out.write("alpha_min: [{}]\n".format(', '.join([str(x) for x in self.alpha_min])))
        yaml_out.write("\n")
        yaml_out.write("N_low_fidelity: {}\n".format(self.N_low_fidelity))
        yaml_out.write("\n")
        yaml_out.close()
        
    def create_model(self):
        raise "Not Implemented"
    
    def forward_cand(self):
        raise "Not Implemented"
    
    def forward_test(self):
        raise "Not Implemented"
    

    ### LINE 5 IN ALGORITHM 1 ###
    # generates candidate solution
    def compute_next_point_cand(self):
        if self.utility_mode == 0:
            return self.compute_next_point_cand_boundary()
        elif self.utility_mode == 1:
            return self.compute_next_point_cand_eic()
        else:
            raise "Not Implemented"
    
    # EXPLORATION
    # Some helper function to generate the candidate solution @ boundary
    def compute_next_point_cand_boundary(self):        
        """
        Evaluates points to determine the next point to sample in the optimization process.
        The method updates the following attributes:
        - self.X_next: The next candidate point to sample.
        - self.X_next_fidelity: The fidelity level of the next candidate point (0 for low-fidelity, 1 for high-fidelity).
        - self.min_time_cand: The minimum time candidate.
        - self.alpha_min_cand: The denormalized candidate point.
        - self.flag_found_ei: A flag indicating whether a point with positive expected improvement was found.
        Returns:
            None
        """
        # Using MFDGP "multi-fideltiy deep gaussian process" to generate GP prior for low fidelity and high fidelity candidates
        # Refer to paragraph after eq (21) in the "Multi-fideltiy black-box optimization for time-optimal quadrotor maneuvers"
        mean_L, var_L, prob_cand_L, prob_cand_L_mean = self.forward_cand()
        # mean_L, var_L, prob_cand_L, mean_H, var_H, prob_cand_H, prob_cand_L_mean = self.forward_cand()

        
        # Equation (24)
        # EXPLORATION
        # initializing points to select the most uncertain sample near the decision boundary later on (will take max to do this later)
        # C represents the cost of an evaluation at fidelity level
        ent_L = -np.abs(mean_L)/(var_L + 1e-9)*self.C_L
        # ent_H = -np.abs(mean_H)/(var_H + 1e-9)*self.C_H
        
        self.flag_found_ei = False
        # max_ei_idx_H = -1
        # max_ei_H = 0
        max_ei_idx_L = -1
        max_ei_L = 0
        min_time_tmp = self.min_time
        for it in range(self.X_cand.shape[0]): #TODO: figure out whats going on here
            x_cand_denorm = self.lb_i + np.multiply(self.X_cand[it,:self.t_dim],self.ub_i-self.lb_i)
            min_time_tmp2 = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
            max_ei_tmp_L = (self.min_time-min_time_tmp2)*prob_cand_L[it]
            # max_ei_tmp_H = (self.min_time-min_time_tmp2)*prob_cand_H[it]
            if max_ei_tmp_L > max_ei_L and prob_cand_L[it] > 1-self.delta_L:
                max_ei_L = max_ei_tmp_L
                max_ei_idx_L = it
            # if max_ei_tmp_H > max_ei_H and prob_cand_H[it] > 1-self.delta_H:
            #     max_ei_H = max_ei_tmp_H
            #     max_ei_idx_H = it
            #     min_time_tmp = min_time_tmp2
        
        X_cand_discard = np.empty(0, dtype=np.int)
        
        # NOTE: im guessing this seciton commented out below is for 
        # expected improvement with constraints

#         if max_ei_idx_L != -1 or max_ei_idx_H != -1:
#             self.flag_found_ei = True
#             if max_ei_H < max_ei_L and self.N_low_fidelity < self.MAX_low_fidelity:
#                 self.X_next = self.X_cand[max_ei_idx_L,:]
#                 self.X_next_fidelity = 0
#             else:
#                 self.X_next = self.X_cand[max_ei_idx_H,:]
#                 X_cand_discard = np.append(X_cand_discard, max_ei_idx_H)
#                 self.X_next_fidelity = 1
#             prPurple("ei_L: {}, ei_H: {}".format(max_ei_L,max_ei_H))
#             self.min_time_cand = min_time_tmp
#         else:
#             if np.max(ent_H) < np.max(ent_L) and self.N_low_fidelity < self.MAX_low_fidelity:
#                 self.X_next = self.X_cand[ent_L.argmax()]
#                 self.X_next_fidelity = 0
#             else:
#                 self.X_next = self.X_cand[ent_H.argmax()]
#                 X_cand_discard = np.append(X_cand_discard, ent_H.argmax())
#                 self.X_next_fidelity = 1
#             prGreen("ent_L: {}, ent_H: {}".format(np.max(ent_L),np.max(ent_H)))
#             x_cand_denorm = self.lb_i + np.multiply(self.X_cand[ent_H.argmax(),:],self.ub_i-self.lb_i)
#             self.min_time_cand = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
        
        # if num low fidelity points < 20 (which is set as the max later)
        # if self.N_low_fidelity < self.MAX_low_fidelity: # TODO figure IF I REALLY NEED TO CHECK MAX FIDELITY BECASE WE ONLY HAVE LOW
        if max_ei_idx_L != -1:
            self.flag_found_ei = True
            self.X_next = self.X_cand[max_ei_idx_L,:]
            self.X_next_fidelity = 0
            prPurple("ei_L: {}".format(max_ei_L))
            self.min_time_cand = min_time_tmp
        # if max_ei_idx_L != -1 or max_ei_idx_H != -1:
        #     self.flag_found_ei = True
        #     if max_ei_H < max_ei_L:
        #         self.X_next = self.X_cand[max_ei_idx_L,:]
        #         self.X_next_fidelity = 0
        #     else:
        #         self.X_next = self.X_cand[max_ei_idx_H,:]
        #         X_cand_discard = np.append(X_cand_discard, max_ei_idx_H)
        #         self.X_next_fidelity = 1
        #     prPurple("ei_L: {}, ei_H: {}".format(max_ei_L,max_ei_H))
        #     self.min_time_cand = min_time_tmp
        else:
            self.X_next = self.X_cand[ent_L.argmax()] ### LINE 6 IN ALGORITHM 1 ###
            self.X_next_fidelity = 0
            # NOTE: max(ent_H) & max(ent_L) select the most uncertain sample near the decision boundary for EXPLORATION
            # if np.max(ent_H) < np.max(ent_L):
            #     self.X_next = self.X_cand[ent_L.argmax()] ### LINE 6 IN ALGORITHM 1 ###
            #     self.X_next_fidelity = 0
            # else:
            #     self.X_next = self.X_cand[ent_H.argmax()] ### LINE 6 IN ALGORITHM 1 ###
            #     X_cand_discard = np.append(X_cand_discard, ent_H.argmax())
            #     self.X_next_fidelity = 1
            # prGreen("ent_L: {}, ent_H: {}".format(np.max(ent_L),np.max(ent_H)))
            prGreen("ent_L: {}".format(np.max(ent_L)))
            # x_cand_denorm = self.lb_i + np.multiply(self.X_cand[ent_H.argmax(),:self.t_dim],self.ub_i-self.lb_i) #TODO FIGURE OUT FORMAT OF X_CAND
            self.min_time_cand = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
        # else: # ELSE IS FOR ALL HIGH FIDELITY STUFF, LOW FIDELITY POINTS EXCEED THE MAX NUM DOES NOT APPLY IF ONLY USING LOW FIDELITY
        #     if max_ei_idx_H != -1:
        #         self.flag_found_ei = True
        #         self.X_next = self.X_cand[max_ei_idx_H,:]
        #         X_cand_discard = np.append(X_cand_discard, max_ei_idx_H)
        #         self.X_next_fidelity = 1
        #         prPurple("ei_H: {}".format(max_ei_H))
        #         self.min_time_cand = min_time_tmp
        #     else:
        #         self.X_next = self.X_cand[ent_H.argmax()] ### LINE 6 IN ALGORITHM 1 ###
        #         X_cand_discard = np.append(X_cand_discard, ent_H.argmax())
        #         self.X_next_fidelity = 1
        #         prGreen("ent_H: {}".format(np.max(ent_H)))
        #         x_cand_denorm = self.lb_i + np.multiply(self.X_cand[ent_H.argmax(),:self.t_dim],self.ub_i-self.lb_i)
        #         self.min_time_cand = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)

        
        self.alpha_min_cand = copy.deepcopy(self.X_next)
        self.alpha_min_cand[:self.t_dim] = self.lb_i + np.multiply(self.X_next[:self.t_dim],self.ub_i-self.lb_i)
        if self.X_next_fidelity == 1:
            print("min time cand: {}, alpha: {}".format(self.min_time_cand, self.alpha_min_cand))

    # EXPLOITATION
    # Helper function to generate next candidate point EIC
    # in exploitation, utilize: "Expected Improvement with Constraints"
    def compute_next_point_cand_eic(self):
        # Using MFDGP "multi-fideltiy deep gaussian process" to generate GP prior for low fidelity and high fidelity candidates
        # Refer to paragraph after eq (21) in the "Multi-fideltiy black-box optimization for time-optimal quadrotor maneuvers"
        mean_L, var_L, prob_cand_L, prob_cand_L_mean = self.forward_cand()
        # mean_L, var_L, prob_cand_L, mean_H, var_H, prob_cand_H, prob_cand_L_mean = self.forward_cand()

        
        # Equation (24)
        # EXPLORATION??
        # TODO: Figure out why these are still calculated in the EXPLOITATION STEP
        # initializing points to select the most uncertain sample near the decision boundary later on (will take max to do this later)
        ent_L = -np.abs(mean_L)/(var_L + 1e-9)*self.C_L
        # ent_H = -np.abs(mean_H)/(var_H + 1e-9)*self.C_H
        
        self.flag_found_ei = False
        # max_ei_idx_H = -1
        # max_ei_H = 0
        max_ei_idx_L = -1
        max_ei_L = 0
        min_time_tmp = self.min_time
        for it in range(self.X_cand.shape[0]):
            x_cand_denorm = self.lb_i + np.multiply(self.X_cand[it,:self.t_dim],self.ub_i-self.lb_i)
            min_time_tmp2 = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
            max_ei_tmp_L = (self.min_time-min_time_tmp2)*prob_cand_L[it]
            # max_ei_tmp_H = (self.min_time-min_time_tmp2)*prob_cand_H[it]
            if max_ei_tmp_L > max_ei_L and prob_cand_L[it] > 1-self.delta_L:
                max_ei_L = max_ei_tmp_L
                max_ei_idx_L = it
            # if max_ei_tmp_H > max_ei_H and prob_cand_H[it] > 1-self.delta_H:
            #     max_ei_H = max_ei_tmp_H
            #     max_ei_idx_H = it
            #     min_time_tmp = min_time_tmp2
        
        ########## No boundary search ###########################################
        max_pb_idx_L = -1
        max_pb_L = 0
        # max_pb_idx_H = -1
        # max_pb_H = 0
        min_time_tmp_pb = self.min_time
        for it in range(self.X_cand.shape[0]):
            x_cand_denorm = self.lb_i + np.multiply(self.X_cand[it,:self.t_dim],self.ub_i-self.lb_i)
            min_time_tmp2 = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
            if self.min_time >= min_time_tmp2 and prob_cand_L[it] > max_pb_L:
                max_pb_L = prob_cand_L[it]
                max_pb_idx_L = it
            # if self.min_time >= min_time_tmp2 and prob_cand_H[it] > max_pb_H:
            #     max_pb_H = prob_cand_H[it]
            #     max_pb_idx_H = it
            #     min_time_tmp_pb = min_time_tmp2
        
        X_cand_discard = np.empty(0, dtype=np.int)
        if max_ei_idx_L != -1: # or max_ei_idx_H != -1: # 
            self.flag_found_ei = True
            # if max_ei_H < max_ei_L and self.N_low_fidelity < self.MAX_low_fidelity:
            self.X_next = self.X_cand[max_ei_idx_L,:]
            self.X_next_fidelity = 0
            # else:
            #     self.X_next = self.X_cand[max_ei_idx_H,:]
            #     X_cand_discard = np.append(X_cand_discard, max_ei_idx_H)
            #     self.X_next_fidelity = 1
            # prPurple("ei_L: {}, ei_H: {}".format(max_ei_L,max_ei_H))
            prPurple("ei_L: {}".format(max_ei_L))
            self.min_time_cand = min_time_tmp
        ########## No boundary search ###########################################
        elif max_pb_idx_L != -1: # or max_pb_idx_H != -1:
            # if max_pb_H < max_pb_L and self.N_low_fidelity < self.MAX_low_fidelity:
            self.X_next = self.X_cand[max_pb_idx_L,:]
            X_cand_discard = np.append(X_cand_discard, max_pb_idx_L)
            self.X_next_fidelity = 0
            # else:
            #     self.X_next = self.X_cand[max_pb_idx_H,:]
            #     X_cand_discard = np.append(X_cand_discard, max_pb_idx_H)
            #     self.X_next_fidelity = 1
            # prPurple("pb_H: {}, pb_L: {}".format(max_pb_H,max_pb_L))
            prPurple("pb_L: {}".format(max_pb_L))
            self.min_time_cand = min_time_tmp_pb
        else:
            # NOTE: max(ent_H) & max(ent_L) select the most uncertain sample near the decision boundary for EXPLORATION
            # if uncertainty of low fidelity point is greater than high and low fidelity points are lower than the max:
            # if np.max(ent_H) < np.max(ent_L) and self.N_low_fidelity < self.MAX_low_fidelity: 
            self.X_next = self.X_cand[ent_L.argmax()] ### LINE 6 IN ALGORITHM 1 ###
            self.X_next_fidelity = 0
            # else: #TODO COMMENT OUT THIS ELSE B/C WE CANT USE HIGH FIDELITY BC WE DONT HAVE
            #     self.X_next = self.X_cand[ent_H.argmax()] ### LINE 6 IN ALGORITHM 1 ###
            #     X_cand_discard = np.append(X_cand_discard, ent_H.argmax())
            #     self.X_next_fidelity = 1
            # prGreen("ent_L: {}, ent_H: {}".format(np.max(ent_L),np.max(ent_H)))
            prGreen("ent_L: {}".format(np.max(ent_L)))
            x_cand_denorm = self.lb_i + np.multiply(self.X_cand[ent_H.argmax(),:self.t_dim],self.ub_i-self.lb_i) # TODO figure out what to change here for candidate stuff
            self.min_time_cand = x_cand_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
        
        self.alpha_min_cand = self.lb_i + np.multiply(self.X_next[:,:self.t_dim],self.ub_i-self.lb_i)
        if self.X_next_fidelity == 1: # TODO (1 means high fidelity), figure out if i can comment this out
            print("min time cand: {}, alpha: {}".format(self.min_time_cand, self.alpha_min_cand))

    # TODO: Figure out what this corresponds to in pseudocode
    def append_next_point(self):
        X_next_denorm = self.lb_i + np.multiply(self.X_next[:self.t_dim],self.ub_i-self.lb_i)
        X_next_time = X_next_denorm.dot(self.t_set_sim)/np.sum(self.t_set_sim)
        print("X_next: {}".format(X_next_denorm))
        print("X_next time: {}".format(X_next_time))
        # if self.X_next_fidelity == 1: #TODO comment out because this is for high fidelty
        #     self.X_H = np.vstack((self.X_H, self.X_next))
        #     Y_next = 1.0*self.sampling_func_H(self.X_next[None,:])
        #     self.Y_H = np.concatenate((self.Y_H, np.array(Y_next)))
        #     self.N_H += 1
        #     if self.min_time > self.min_time_cand and Y_next >= 1:
        #         self.min_time = self.min_time_cand
        #         self.alpha_min = self.alpha_min_cand
        #         prYellow("min time: {}, alpha: {}".format(self.min_time, self.alpha_min))
        #     self.N_low_fidelity = 0
        # else:
        #     self.N_low_fidelity += 1
        #     print("low fidelity: {}/{}".format(self.N_low_fidelity,self.MAX_low_fidelity))
        #     self.X_L = np.vstack((self.X_L, self.X_next))
        #     Y_next = 1.0*self.sampling_func_L(self.X_next[None,:])
        #     self.Y_L = np.concatenate((self.Y_L, np.array(Y_next)))
        #     self.N_L += 1
        self.N_low_fidelity += 1
        print("low fidelity: {}/{}".format(self.N_low_fidelity,self.MAX_low_fidelity))
        self.X_L = np.vstack((self.X_L, self.X_next))
        Y_next = 1.0*self.sampling_func_L(self.X_next[None,:])
        self.Y_L = np.concatenate((self.Y_L, np.array(Y_next)))
        self.N_L += 1
        self.exp_result_array.append(Y_next[0])
        # rel_snap = 1.0*self.sampling_func_H(self.X_next[None,:], return_snap=True)
        # self.rel_snap_array.append(rel_snap[0])
        # print("rel_snap: {}".format(rel_snap[0]))
#         if rel_snap[0] < 0.999:
#             prRed("Wrong rel snap: {}".format(rel_snap[0]))
#             prRed("X_next_denorm: {}".format(X_next_denorm))
#             raise("ERROR REL SNAP")
            
        # print("N_L: {}, N_H: {}".format(self.N_L, self.N_H))
        print("N_L: {}".format(self.N_L))
        
        if self.X_cand.shape[0] < self.N_cand:
            print("Remaining X_cand: {}".format(self.X_cand.shape[0]))
            self.X_cand = np.append(self.X_cand, self.sample_data(self.N_cand-self.X_cand.shape[0]),0)
        print("-------------------------------------------")
    
    def save_result_data(self, filedir, filename_result):
        yamlFile = os.path.join(filedir, filename_result)
        yaml_out = open(yamlFile,"w")
        high_idx = 0
        low_idx = 0
        for it in range(len(self.min_time_array)):
            if self.fidelity_array[it] == 1:
                yaml_out.write("iter{}:\n".format(high_idx))
                high_idx += 1
                low_idx = 0
            else:
                yaml_out.write("iter{}_{}:\n".format(high_idx-1,low_idx))
                low_idx += 1
            yaml_out.write("  found_ei: {}\n".format(self.found_ei_array[it]))
            yaml_out.write("  exp_result: {}\n".format(self.exp_result_array[it]))
            yaml_out.write("  rel_snap: {}\n".format(self.rel_snap_array[it]))
            yaml_out.write("  min_time: {}\n".format(self.min_time_array[it]))
            yaml_out.write("  alpha_cand: [{}]\n\n".format(','.join([str(x) for x in self.alpha_cand_array[it]])))
        yaml_out.close()

    ### LINES 5-7 IN ALGORITHM 1 ###
    def active_learning(self, N=15, MAX_low_fidelity=20, plot=False, filedir='./mfbo_data', \
                        filename_plot='active_learning_%i.png', \
                        filename_result='result.yaml', \
                        filename_exp='exp_data.yaml'):
        
        if not hasattr(self, 'start_iter'):
            self.start_iter = 0
            self.min_time_array = []
            self.alpha_cand_array = []
            self.fidelity_array = []
            self.found_ei_array = []
            self.exp_result_array = []
            self.rel_snap_array = []
            self.min_time_array.append(self.min_time)
            self.alpha_cand_array.append(self.alpha_min_cand)
            self.exp_result_array.append(1)
            self.rel_snap_array.append(1)
            self.fidelity_array.append(1)
            self.found_ei_array.append(1)
            self.writer.add_scalar('/min_time', 1.0, 0)
            self.writer.add_scalar('/num_low_fidelity', 0, 0)
            self.writer.add_scalar('/num_found_ei', 0, 0)
            self.writer.add_scalar('/num_failure', 0, 0)
            self.writer.add_scalar('/rel_snap', 1.0, 0)
        
        self.MAX_low_fidelity = MAX_low_fidelity
        main_iter_start = self.start_iter
        self.min_time = self.min_time_array[-1]
        
        # Save results if the starting iteration is the last one
        if main_iter_start == N-1:
            self.save_result_data(filedir, filename_result)
        # Main loop for active learning iterations

        for main_iter in range(main_iter_start, N):
            prGreen("#################################################")
            print('%i / %i' % (main_iter+1,N))
            self.X_next_fidelity = 0
            if not hasattr(self, 'N_low_fidelity'):
                self.N_low_fidelity = 0
            # If the next point is not found, create a model and compute the next point
            num_found_ei = 0
                    # Create a model
            num_low_fidelity = self.N_low_fidelity
                    # Compute the next point
            while self.X_next_fidelity == 0:
                try:
                    self.create_model(num_epochs=self.iter_create_model)
                    self.compute_next_point_cand()
                except RuntimeError as e:
                    # if 'out of memory' in str(e):
                    #     print('| WARNING: ran out of memory, retrying batch')
                    #     for p in self.model.parameters():
                    #         if p.grad is not None:
                    #             del p.grad  # free some memory
                    #     torch.cuda.empty_cache()
                    #     self.create_model(num_epochs=self.iter_create_model)
                    #     self.compute_next_point_cand()
                    # elif 'cholesky_cuda' in str(e):
                    #     print('| WARNING: cholesky_cuda')                        
                    #     if hasattr(self, 'clf'):
                    #         del self.clf                        
                    #     if hasattr(self, 'feature_model'):
                    #         del self.feature_model
                    #     self.create_model()
                    #     self.compute_next_point_cand()
                    # else:
                    raise e
                # Append the next point
                self.append_next_point()
                if plot:
                    prefix = self.model_prefix.split("/")[1]+"_"+str(self.rand_seed)
                    filepath = os.path.join(filedir,prefix)
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    filepath = os.path.join(filepath,filename_plot%main_iter)
                    self.plot(filename = filepath)

                self.min_time_array.append(self.min_time)
                self.alpha_cand_array.append(self.alpha_min_cand)
                self.fidelity_array.append(self.X_next_fidelity)
                if self.flag_found_ei:
                    self.found_ei_array.append(1)
                    num_found_ei += 1
                else:
                    self.found_ei_array.append(0)
                num_low_fidelity += 1
                
                if self.X_next_fidelity == 0:
                    self.start_iter = main_iter
                else:
                    self.start_iter = main_iter+1
                self.save_exp_data(filedir, filename_exp)

            num_failure = 0
            for it in range(len(self.min_time_array)):
                if self.fidelity_array[it] == 1 and self.exp_result_array[it] == 0:
                    num_failure += 1
            self.writer.add_scalar('/min_time', self.min_time, main_iter+1)
            self.writer.add_scalar('/num_low_fidelity', num_low_fidelity, main_iter+1)
            self.writer.add_scalar('/num_found_ei', num_found_ei, main_iter+1)
            self.writer.add_scalar('/num_failure', num_failure, main_iter+1)
            
            min_time_idx = 0
            for it in range(len(self.min_time_array)):
                if self.fidelity_array[it] == 1 and self.min_time_array[it] == self.min_time:
                    min_time_idx = it
                    break
            self.writer.add_scalar('/rel_snap', self.rel_snap_array[min_time_idx], main_iter+1)
        
            self.save_result_data(filedir, filename_result)
        return

    def plot(self, filename='MFBO_2D.png'):
        assert self.dim == 2
        mean, var, prob_cand = self.forward_test()
        
        ent_cand = -np.abs(mean)/(var + 1e-9)

        fig = plt.figure(1,figsize=(9,7.5))
        plt.clf()
        ax = plt.subplot(111)
        
        X_test_denorm_ = np.repeat(np.expand_dims(self.lb_i,0),self.X_test.shape[0],axis=0) + np.multiply(self.X_test, np.repeat(np.expand_dims(self.ub_i-self.lb_i,0),self.X_test.shape[0],axis=0))
        X_test_time_ = np.multiply(X_test_denorm_, np.repeat(np.expand_dims(self.t_set_sim,0),self.X_test.shape[0],axis=0))
        cnt = plt.tricontourf(X_test_time_[:,0], X_test_time_[:,1], prob_cand, np.linspace(-0.01,1.01,100),cmap='coolwarm_r',alpha=0.4)
        for c in cnt.collections:
            c.set_edgecolor("face")

        cb = plt.colorbar(ticks = [0,1])
        cb.set_label('feasibility (y)', labelpad=0, fontsize='xx-large')

        cb.ax.tick_params(labelsize='xx-large')
        labels = cb.ax.get_yticklabels()
        labels[0].set_verticalalignment("bottom")
        labels[-1].set_verticalalignment("top")

        X_L_denorm_ = np.repeat(np.expand_dims(self.lb_i,0),self.X_L.shape[0],axis=0) \
            + np.multiply(self.X_L,np.repeat(np.expand_dims(self.ub_i-self.lb_i,0),self.X_L.shape[0],axis=0))
        X_L_time_ = np.multiply(X_L_denorm_, np.repeat(np.expand_dims(self.t_set_sim,0),X_L_denorm_.shape[0],axis=0))
        plt.scatter(X_L_time_[:,0], X_L_time_[:,1],c=self.Y_L[:],cmap='coolwarm_r', \
                    marker='x',s=50,label='low fidelity sample')

        # X_H_denorm_ = np.repeat(np.expand_dims(self.lb_i,0),self.X_H.shape[0],axis=0) + np.multiply(self.X_H, np.repeat(np.expand_dims(self.ub_i-self.lb_i,0),self.X_H.shape[0],axis=0))
        # X_H_time_ = np.multiply(X_H_denorm_, np.repeat(np.expand_dims(self.t_set_sim,0),X_H_denorm_.shape[0],axis=0))
        # plt.scatter(X_H_time_[:,0], X_H_time_[:,1],c=self.Y_H[:],cmap='coolwarm_r',\
        #             s=150,edgecolors='k',label='high fidelity sample')
        
        # X_best_denorm_ = lb_i + np.multiply(X_best, ub_i-lb_i)
        X_best_time_ = np.multiply(self.alpha_min, self.t_set_sim)
        plt.scatter([X_best_time_[0]],[X_best_time_[1]],color='lawngreen', \
                    marker='*',edgecolors='k',s=400,label='current best solution')
        
#         for k in range(self.N_H-self.N_H_i):
#             plt.text(X_H_time_[self.N_H_i+k,0], X_H_time_[self.N_H_i+k,1], str(k+1), fontsize=12, color='green')
        plt.legend()
        
        X_next_denorm_ = self.lb_i + np.multiply(self.X_next, self.ub_i-self.lb_i)
        X_next_time_ = np.multiply(X_next_denorm_, self.t_set_sim)
        
        plt.scatter([X_next_time_[0]],[X_next_time_[1]],color = 'k',marker = '*')
        plt.legend(loc=4, fontsize='xx-large')
        plt.xlim([self.lb_i[0]*self.t_set_sim[0],self.ub_i[0]*self.t_set_sim[0]])
        plt.ylim([self.lb_i[1]*self.t_set_sim[1],self.ub_i[1]*self.t_set_sim[1]])
        
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        def prettyplot_tmp(xlabel, ylabel, xlabelpad = -10, ylabelpad = -20, minXticks = True, minYticks = True):
            plt.xlabel(xlabel, labelpad = xlabelpad, fontsize='xx-large')
            plt.ylabel(ylabel, labelpad = ylabelpad, fontsize='xx-large')

            if minXticks:
                plt.xticks(plt.xlim(), fontsize='xx-large')
                rang, labels = plt.xticks()
                labels[0].set_horizontalalignment("left")
                labels[-1].set_horizontalalignment("right")

            if minYticks:
                plt.yticks(plt.ylim(), fontsize='xx-large')
                rang, labels = plt.yticks()
                labels[0].set_verticalalignment("bottom")
                labels[-1].set_verticalalignment("top")

        prettyplot_tmp("$\mathregular{x_1}$ [s]", "$\mathregular{x_2}$ [s]", ylabelpad=-15)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

class ActiveMFDGP(MFBOAgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.min_loss = -1
        self.batch_size = kwargs.get('gpu_batch_size', 256)

    def create_model(self, num_epochs=500):
        self.train_x_L = torch.tensor(self.X_L).float()#.cuda()
        self.train_y_L = torch.tensor(self.Y_L).float()#.cuda()
        self.train_dataset_L = TensorDataset(self.train_x_L, self.train_y_L)
        self.train_loader_L = DataLoader(self.train_dataset_L, batch_size=self.batch_size, shuffle=True)

        # self.train_x_H = torch.tensor(self.X_H).float()#.cuda()
        # self.train_y_H = torch.tensor(self.Y_H).float()#.cuda()
        # self.train_dataset_H = TensorDataset(self.train_x_H, self.train_y_H)
        # self.train_loader_H = DataLoader(self.train_dataset_H, batch_size=self.batch_size, shuffle=True)
        
        train_x = [self.train_x_L]
        # train_x = [self.train_x_L, self.train_x_H]

        train_y = [self.train_y_L]
        # train_y = [self.train_y_L, self.train_y_H]
        
        if not hasattr(self, 'clf'):
            self.clf = MFDeepGPC(train_x, train_y, num_inducing=128)#.cuda()

        optimizer = torch.optim.Adam([
            {'params': self.clf.parameters()},
        ], lr=0.001)
        mll = VariationalELBO(self.clf.likelihood, self.clf, self.train_x_L.shape[-2])
        # mll = VariationalELBO(self.clf.likelihood, self.clf, self.train_x_L.shape[-2]+self.train_x_H.shape[-2])
        start_time = time.time()
        N_data = self.X_L.shape[0]
        # N_data = self.X_L.shape[0] + self.X_H.shape[0]

        with gpytorch.settings.fast_computations(log_prob=False, solves=False):
            for i in range(num_epochs):
                avg_loss = 0
                for minibatch_i, (x_batch, y_batch) in enumerate(self.train_loader_L):
                    optimizer.zero_grad()
                    output = self.clf(x_batch, fidelity=1)
                    loss = -mll(output, y_batch)
                    loss.backward(retain_graph=True)
                    avg_loss += loss.item()/N_data
                    optimizer.step()

                # for minibatch_i, (x_batch, y_batch) in enumerate(self.train_loader_H):
                #     optimizer.zero_grad()
                #     output = self.clf(x_batch, fidelity=2)
                #     loss = -mll(output, y_batch)
                #     output_L = self.clf(x_batch, fidelity=1)
                #     loss -= mll(output_L, y_batch)
                #     avg_loss += loss.item()/N_data
                #     loss.backward(retain_graph=True)
                #     optimizer.step()

                if (i+1)%20 == 0 or i == 0:
                    print('Epoch %d/%d - Loss: %.3f' % (i+1, num_epochs, avg_loss))
                
                if self.min_loss > avg_loss and (i+1) >= 20:
                    print('Early stopped at Epoch %d/%d - Loss: %.3f' % (i+1, num_epochs, avg_loss))
                    break
        
        if self.min_loss < 0:
            self.min_loss = avg_loss
        
        print(" - Time: %.3f" % (time.time() - start_time))
    
    # line 8 of Algorithm 1 (i think)
    def forward_cand(self):
        self.X_cand = self.sample_data(self.N_cand)
        if self.sampling_mode >= 2:
            self.X_cand[:,:self.t_dim] = np.multiply(self.X_cand[:,:self.t_dim], \
                                       np.repeat(np.expand_dims(self.alpha_min[:self.t_dim],0),self.X_cand.shape[0],axis=0))
            self.X_cand[:,:self.t_dim] += self.lb_i/(self.ub_i-self.lb_i)*(self.alpha_min[:self.t_dim]-1)
            self.X_cand = self.X_cand[(np.min(self.X_cand[:,:self.t_dim]-self.lb_i,axis=1)>=0) \
                                                     & (np.max(self.X_cand[:,:self.t_dim]-self.ub_i,axis=1)<=0),:]
        
        test_x = torch.tensor(self.X_cand).float()#.cuda()
        test_dataset = TensorDataset(test_x)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        mean_L = np.empty(0)
        var_L = np.empty(0)
        prob_cand_L = np.empty(0)
        prob_cand_L_mean = np.empty(0)
        # mean_H = np.empty(0)
        # var_H = np.empty(0)
        # prob_cand_H = np.empty(0)
        
        for minibatch_i, (x_batch,) in enumerate(test_loader): #TODO
            p, m, v, pm = self.clf.predict_proba_MF(x_batch, fidelity=1, C_L=self.C_L, beta=self.beta, return_all=True)
            # p, m, v, pm = self.clf.predict_proba_MF(x_batch, fidelity=1, C_H=self.C_H, C_L=self.C_L, beta=self.beta, return_all=True)

            mean_L = np.append(mean_L, m)
            var_L = np.append(var_L, v)
            prob_cand_L = np.append(prob_cand_L, p[:,1])
            prob_cand_L_mean = np.append(prob_cand_L_mean, pm[:,1])
        
            # p_H, m_H, v_H, pm_H = self.clf.predict_proba_MF(x_batch, fidelity=2, C_H=self.C_H, C_L=self.C_L, beta=self.beta, return_all=True)
            # p_H, m_H, v_H, pm_H = self.clf.predict_proba_MF(x_batch, fidelity=2, C_H=self.C_H, C_L=self.C_L, beta=self.beta, return_all=True)
            # mean_H = np.append(mean_H, m_H)
            # var_H = np.append(var_H, v_H)
            # prob_cand_H = np.append(prob_cand_H, p_H[:,1])
        
        return mean_L, var_L, prob_cand_L, prob_cand_L_mean
        # return mean_L, var_L, prob_cand_L, mean_H, var_H, prob_cand_H, prob_cand_L_mean

    
    def forward_test(self):
        test_x_L = torch.tensor(self.X_test).float()#.cuda()
        test_dataset_L = TensorDataset(test_x_L)
        test_loader_L = DataLoader(test_dataset_L, batch_size=self.batch_size, shuffle=False)

        mean = np.empty(0)
        var = np.empty(0)
        prob_cand = np.empty(0)
        for minibatch_i, (x_batch,) in enumerate(test_loader_L):
            m, v, pm = self.clf.predict_proba(x_batch, fidelity=2, return_std=False)
            mean = np.append(mean, m)
            var = np.append(var, v)
            prob_cand = np.append(prob_cand, pm[:,1])
        
        return mean, var, prob_cand

