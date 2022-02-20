#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import randint, uniform


class EpsilonGreedy:
    def __init__(self, epsilon, controls):
        self.epsilon = epsilon  # probability to act randomly
        self.controls = controls  # all possible controls for joints

    def __call__(self, state, Q, log=False):

        if(uniform() < self.epsilon):
            control = list()
            for joint_control in self.controls:
                control.append(joint_control[randint(0, len(joint_control))])
            if log:
                print("Random control", control)
            return control

        return self.optimal(state, Q, log)

    def optimal(self, x, Q, log=False):
        control = list()
        inputs = np.array([x])
        outputs = Q(inputs, training=False)
        index = np.argmin(outputs)
        if len(self.controls) == 1:
            control.append(index)
        else:
            control.append(index % len(self.controls[0]))
            control.append(int(np.floor(index/len(self.controls[1]))))
        if log:
            print("Control", control)
        return control
