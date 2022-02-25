import os.path

import tensorflow as tf
# from tensorflow.keras import layers
# for some reason in this build the import above gives the error:
# cannot find reference 'keras' in '__init__.py'
from keras import layers
import numpy as np
from dpendulum import DPendulum
from buffer import ExperienceReplay
from policy import EpsilonGreedy
import matplotlib.pyplot as plt

from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(nx)
    state_out1 = layers.Dense(16, activation="relu")(inputs)
    state_out2 = layers.Dense(32, activation="relu")(state_out1)
    state_out3 = layers.Dense(64, activation="relu")(state_out2)
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(nu)(state_out4)
    model = tf.keras.Model(inputs, outputs)
    return model


def update(x_batch, u_batch, cost_batch, x_next_batch, Q_target, Q,
           critic_optimizer, discount_factor, nu):
    """
    Update the weights of the Q network using the
    specified batch of data
    """
    # all inputs are tf tensors
    with tf.GradientTape() as tape:

        target_output = Q_target(x_next_batch, training=True)
        target_values = np.min(target_output, 1, keepdims=True)
        # Compute 1-step targets for the critic loss
        y = cost_batch + discount_factor*target_values
        # Compute batch of Values associated to the sampled batch of states
        Q_outputs = Q(x_batch, training=True)
        selection = np.arange(len(Q_outputs))
        if len(u_batch[0]) > 1:
            u_b = [u[0]+u[1]*nu for u in u_batch]
        else:
            u_b = np.ndarray.flatten(u_batch)

        Q_values = Q_outputs[selection, u_b]
        batch_Q_values = np.reshape(Q_values, (-1, 1))
        # Critic's loss function. tf.math.reduce_mean() computes the mean of
        # elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - batch_Q_values))

    # Compute the gradients of the critic loss w.r.t. critic's parameters
    # (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)

    # stabilizer used in the paper
    Capped_Q_grad = [tf.clip_by_value(g, -1, 1) for g in Q_grad]

    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Capped_Q_grad, Q.trainable_variables))

    return Q_loss


if __name__ == "__main__":

    NUMBER_OF_JOINTS = 1

    WEIGHTS_FILE_PATH = os.path.abspath("nn_weights.h5")

    EPISODES = 300
    EPISODE_LENGHT = 2**8

    EXPERIENCE_REPLAY_SIZE = 2**16
    BATCH_SIZE = 2**6
    NO_OP_THRESHOLD = 2**14
    DISCOUNT_FACTOR = 0.95

    EPSILON = 1
    EPSILON_MAX = 1
    EPSILON_MIN = 0.001
    EPSILON_DECAY = 0.008 * 500
    # the target network is updated every N gradient descent
    TARGET_UPDATE_THRESHOLD = 2**6
    # the number of steps to execute between each gradient descent
    GRADIENT_DESCENT_THRESHOLD = 4
    # step skipping, number of steps passed between the selection of a different action
    ACTION_SELECTION_THRESHOLD = 1
    # the best network is saved once every N episode
    SAVE_NETWORK_THRESHOLD = 15

    LEARNING_RATE = 0.0001

    pendulum = DPendulum(joints=NUMBER_OF_JOINTS)
    target_update = 0
    gradients_update = 0
    action_selection = 0
    save_network = 0

    # Experience replay initialization
    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)

    # Policy initialization
    policy = EpsilonGreedy(EPSILON, pendulum.controls())

    # Create critic and target NNs
    nx = pendulum.nx
    nu = pendulum.nu
    nq = pendulum.nq
    Q = get_critic(nx, nu**nq)
    Q_target = get_critic(nx, nu**nq)

    # Set initial weights of targets equal to those of actor and critic
    Q_target.set_weights(Q.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    data = {}

    # filling the experience replay buffer
    buffer.fill(NO_OP_THRESHOLD, EPISODE_LENGHT, pendulum, policy, Q, 4)

    target_update = 0
    gradients_update = 0
    action_selection = 0
    save_network = 0
    average_cost_to_go = 0
    best_average_cost_to_go = np.Inf

    # Training

    for e in range(EPISODES):
        x = pendulum.reset()
        u = pendulum.c2du(np.zeros(pendulum.nq))

        data[e] = {'loss': [], 'x': [], 'next_x': []}
        cost_to_go = 0
        discount = 1

        with tqdm(total=EPISODE_LENGHT) as pbar:
            pbar.set_description('Episode %d' % (e+1))
            for i in range(EPISODE_LENGHT):
                logs = True if e+1 == EPISODES else False

                x = pendulum.x.copy()

                action_selection += 1
                if action_selection > ACTION_SELECTION_THRESHOLD:
                    u = policy(x, Q)
                    action_selection = 1

                next_x, cost = pendulum.step(u)

                cost_to_go += discount*cost
                discount *= DISCOUNT_FACTOR

                buffer.add_transition(x=x, u=u, cost=cost, next_x=next_x)

                if logs:
                    print(i)
                    print("x", x)
                    print("u", u)
                    print("cost", cost)
                    print("")

                data[e]['x'].append(x[0].copy())
                data[e]['next_x'].append(next_x[0].copy())

                gradients_update += 1
                if gradients_update > GRADIENT_DESCENT_THRESHOLD:
                    batch = buffer.sample(BATCH_SIZE)
                    x_batch = np.array([b[0] for b in batch])
                    u_batch = np.array([b[1] for b in batch])
                    cost_batch = np.array([b[2]
                                          for b in batch]).reshape((-1, 1))
                    x_next_batch = np.array([b[3] for b in batch])

                    _loss = update(x_batch, u_batch, cost_batch, x_next_batch,
                                   Q_target, Q, critic_optimizer,
                                   DISCOUNT_FACTOR, pendulum.nu)
                    gradients_update = 1
                    data[e]['loss'].append(_loss)  # Takes the float value

                target_update += 1
                if target_update > TARGET_UPDATE_THRESHOLD:
                    Q_target.set_weights(Q.get_weights())
                    target_update = 1

                pbar.update(1)

                if e + 1 == EPISODES:
                    pendulum.render()

            pbar.close()

        print("Cost to go:", cost_to_go)
        save_network += 1
        average_cost_to_go += cost_to_go
        if save_network > SAVE_NETWORK_THRESHOLD:
            average_cost_to_go /= SAVE_NETWORK_THRESHOLD
            if average_cost_to_go < best_average_cost_to_go:
                print("Saving network with {} average cost to go"
                      .format(average_cost_to_go))
                Q.save_weights(WEIGHTS_FILE_PATH)
                best_average_cost_to_go = average_cost_to_go
            average_cost_to_go = 0
            save_network = 1

        proportion = e / EPISODES
        EPSILON = max(EPSILON_MIN, np.exp(-EPSILON_DECAY*proportion))
        print("Epsilon", EPSILON)
        policy.epsilon = EPSILON
        print("")

    pendulum.reset()
    episode_cost = 0
    discount = 1
    Q.load_weights(WEIGHTS_FILE_PATH)
    for i in range(EPISODE_LENGHT):
        x = pendulum.x.copy()
        u = policy.optimal(x, Q)
        next_x, cost = pendulum.step(u)
        cost += discount * cost
        discount *= DISCOUNT_FACTOR
        episode_cost += cost
        pendulum.render()
    print("Final episode cost", episode_cost)
