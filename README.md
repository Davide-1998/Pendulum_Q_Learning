# Pendulum_Q_Learning

A simple project showcasing the use of Q-learning for balancing a double pendulum.

## Description

This repository contains the code necessary to run a Q-Learning algorithm ruling the behaviour of a pendulum.
It focuses on the adaptation of the work in the paper [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236) to a simpler environment, along with many suggestion from [Implementing the Deep Q-Network](https://arxiv.org/abs/1711.07478) for implementing the details. The resulting deep neural network is able to accurately estimate the Q-function of the pendulum along with a greedy policy which together allow it to produce episodes with low cost, managing to balance the pendulum starting from any state.

## Getting Started

### Dependencies

* Python3
* Tensorflow
* Keras
* Numpy
* Orca

### Executing program

 It is sufficient to execute the DQN.py script

## Authors

[Davide-1998](https://github.com/Davide-1998)

[LazyTurtle](https://github.com/LazyTurtle)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
