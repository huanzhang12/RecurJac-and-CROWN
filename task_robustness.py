## task_robustness.py
## 
## Robustness verification/certification for neural networks
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import time
import numpy as np
from utils import binary_search
from bound_base import compute_bounds, compute_bounds_integral
from bound_spectral import spectral_bound

class task(object):
    def __init__(self, **kwargs):
        # add all arguments
        for k, v in kwargs.items():
            if not k.startswith("__"):
                exec('self.{} = kwargs["{}"]'.format(k, k))
        self.Nsamp = 0
        self.robustness_lb_sum = 0.0
        # not force label
        assert not self.args.targettype.isdecimal()
        print("starting robustness verification on {} images!".format(len(self.inputs)))

    def warmup(self, **kwargs):
        args = self.args
        if args.layerbndalg == "spectral":
            spectral_bound(self.weights, self.biases, 0, 1, self.inputs[0], self.preds[0], self.numlayer, self.activation, args.norm, not self.targeted)
        else:
            compute_bounds(self.weights, self.biases, 0, 1, self.inputs[0], self.preds[0], self.numlayer,args.norm, 0.01, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param)

    def run_single(self, i):
        args = self.args
        weights = self.weights
        biases = self.biases
        inputs = self.inputs
        preds = self.preds
        eps = args.eps

        self.Nsamp += 1
        predict_label = np.argmax(self.true_labels[i])
        target_label = np.argmax(self.targets[i])
        start = time.time()
        # Spectral bound: no binary search needed
        if args.layerbndalg == "spectral":
            robustness_lb, _ = spectral_bound(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, self.activation, args.norm, not self.targeted)
        elif args.jacbndalg != "disable":
            def binary_search_cond(current_eps):
                robustness_lb = compute_bounds_integral(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, current_eps, args.lipsteps, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
                return robustness_lb == current_eps, robustness_lb
            # perform binary search
            robustness_lb = binary_search(binary_search_cond, eps)

            if False:
                # Using local Lipschitz constant to verify robustness. 
                # adaptively find a good eps
                robustness_lb = compute_bounds_integral(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, args.eps, args.lipsteps, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
                # if initial eps is too small, then increase it
                max_robustness_lb = robustness_lb
                if robustness_lb == eps:
                    while robustness_lb == eps:                   
                        eps = eps*2
                        print("*** eps too small, increase eps to {} ***".format(eps))
                        robustness_lb = compute_bounds_integral(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, eps, args.lipsteps, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
                        max_robustness_lb = max(robustness_lb, max_robustness_lb)
                # if initial eps is too large, then decrease it
                elif robustness_lb <= eps / 5:
                    # this makes eps decrease quickly
                    while robustness_lb <= eps / 5:
                        eps = eps/5
                        print("*** eps too large, decrease eps to {} ***".format(eps))
                        robustness_lb = compute_bounds_integral(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, eps, args.lipsteps, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
                        max_robustness_lb = max(robustness_lb, max_robustness_lb)
                    # after decreasing it, try to increase it a little bit
                    while robustness_lb == eps:             
                        eps = eps*2
                        print("*** final adjustment on eps, increase eps to {} ***".format(eps))
                        robustness_lb = compute_bounds_integral(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, eps, args.lipsteps, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
                        max_robustness_lb = max(robustness_lb, max_robustness_lb)
                print('max_robustness_lb = {}, robustness_lb = {}'.format(max_robustness_lb, robustness_lb))
                robustness_lb = max_robustness_lb
        else:
            # use linear outer bounds to verify robustness
            def binary_search_cond(current):
                gap_gx, _, _, _= compute_bounds(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, current, args.layerbndalg, "disable", untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param)
                return gap_gx >=0, gap_gx
            
            # perform binary search
            robustness_lb = binary_search(binary_search_cond, eps)

        self.robustness_lb_sum += robustness_lb
        # get the gradient at this data point
        gradients = self.model.get_gradient(inputs[i:i+1])
        obj_grad = gradients[predict_label] - gradients[target_label]
        q = int(1.0/ (1.0 - 1.0/args.norm)) if args.norm != 1 else np.inf
        grad_norm = np.linalg.norm(obj_grad.flatten(), ord = q)
        predictions = self.model.model.predict(inputs[i:i+1])
        margin = predictions[0][predict_label] - predictions[0][target_label]
        print("[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {}, info = {}, robustness_lb = {:.5f}, avg_robustness_lb = {:.5f}, margin = {:.4f}, grad_norm = {:.4f}, time = {:.4f}".format(self.modelfile, i, self.true_ids[i], predict_label, target_label, self.img_info[i], robustness_lb, self.robustness_lb_sum/self.Nsamp, margin, grad_norm, time.time() - start))

    def summary(self, **kwargs):
        print("[L0] model = {}, avg robustness_lb = {:.5f}, numimage = {}".format(self.modelfile,self.robustness_lb_sum/self.Nsamp,self.Nsamp))

