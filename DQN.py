import os.path

import tensorflow as tf
# from tensorflow.keras import layers
# for some reason in this build the import above gives the error:
# cannot find reference 'keras' in '__init__.py'
from keras import layers
import numpy as np
from orca.orca import init

from manipulator.dpendulum import DPendulum
from buffer import ExperienceReplay
from policy import EpsilonGreedy

from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def get_critic(state_cardinality, possible_actions):
    """ Create the neural network to represent the Q function """
    inputs = layers.Input(state_cardinality)
    state_out1 = layers.Dense(16, activation="relu")(inputs)
    state_out2 = layers.Dense(32, activation="relu")(state_out1)
    state_out3 = layers.Dense(64, activation="relu")(state_out2)
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(possible_actions)(state_out4)
    model = tf.keras.Model(inputs, outputs)
    return model


def update(states_batch, controls_batch, costs_batch, next_states_batch,
           is_state_final_batch, Q_target, Q, optimizer, discount_factor,
           controls):
    """
    Update the weights of the Q network using the
    specified batch of data
    """
    # all inputs are tf tensors
    with tf.GradientTape() as tape:

        target_output = Q_target(next_states_batch, training=True)
        target_values = np.array(np.min(target_output, 1, keepdims=True))

        target_values[is_state_final_batch] = 0
        # Compute 1-step targets for the critic loss
        y = costs_batch + discount_factor * target_values
        # Compute batch of Values associated to the sampled batch of states
        Q_outputs = Q(states_batch, training=True)
        selection = np.arange(len(Q_outputs))

        if len(controls_batch[0]) > 1:
            # if there are 2 joints, discretize the action representation
            u_b = [item[0] + item[1] * controls for item in controls_batch]
        else:
            u_b = np.ndarray.flatten(controls_batch)

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

    # Update the critic back propagating the gradients
    optimizer.apply_gradients(zip(Capped_Q_grad, Q.trainable_variables))

    return Q_loss


def test_network(robot, Q, pi, initial_state=None, episode_length=256,
                 record_namefile='', record_folder=os.getcwd()):
    robot.reset(initial_state)
    episode_cost = 0
    discount_factor = 1
    robot.pendulum.record_pendulum(custom_namefile=record_namefile,
                                   movie_dir=record_folder)

    for _ in range(episode_length):
        state = robot.x.copy()
        control = pi.optimal(state, Q)
        next_state, step_cost = robot.step(control)
        episode_cost += discount_factor * step_cost
        discount_factor *= DISCOUNT_FACTOR
        robot.render()

    robot.pendulum.end_record(record_namefile, record_folder)
    return episode_cost


if __name__ == "__main__":

    #####################################
    # Custom global parameters
    #####################################

    MOVIE_DIR = os.getcwd() + os.sep + 'Movie'

    #####################################
    # Hyper-parameters
    #####################################

    NUMBER_OF_JOINTS = 2
    # the number of quantization levels for controls should be an odd number
    QUANTIZATION_LEVELS = 15

    BEST_WEIGHTS_FILE_PATH = os.path.abspath("weights/best_network_weights.h5")
    TRAINED_WEIGHTS_FILE_PATH = os.path.abspath(
        "weights/last_network_weights.h5")

    EPISODES = 1 # 500
    EPISODE_LENGTH = 1 # 2 ** 8

    EXPERIENCE_REPLAY_SIZE = 2 ** 16
    BATCH_SIZE = 2 ** 6
    NO_OP_THRESHOLD = 2 ** 14
    DISCOUNT_FACTOR = 0.95

    EPSILON = 1
    EPSILON_MAX = 1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = - 0.011 * 500
    # the target network is updated every N gradient descent
    TARGET_UPDATE_THRESHOLD = 2 ** 6
    # the number of steps to execute between each gradient descent
    GRADIENT_DESCENT_THRESHOLD = 4
    # step skipping, number of steps passed
    # between the selection of a different action
    ACTION_SELECTION_THRESHOLD = 1
    # the best network is saved once every N episode
    SAVE_NETWORK_THRESHOLD = 15

    LEARNING_RATE = 0.0001

    #####################################

    pendulum = DPendulum(joints=NUMBER_OF_JOINTS, nu=QUANTIZATION_LEVELS)
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
    Q_network = get_critic(nx, nu ** nq)
    Q_network_target = get_critic(nx, nu ** nq)

    # Set initial weights of targets equal to those of actor and critic
    Q_network_target.set_weights(Q_network.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    data = {}

    # filling the experience replay buffer
    buffer.fill(NO_OP_THRESHOLD, EPISODE_LENGTH,
                pendulum, policy, Q_network, 2)

    average_cost_to_go = 0
    best_average_cost_to_go = np.Inf

    # Training

    for e in range(EPISODES):
        x = pendulum.reset()
        u = pendulum.c2du(np.zeros(pendulum.nq))

        data[e] = {'loss': [], 'x': [], 'next_x': []}
        cost_to_go = 0
        discount = 1

        with tqdm(total=EPISODE_LENGTH) as pbar:
            pbar.set_description('Episode %d' % (e + 1))
            for i in range(EPISODE_LENGTH):

                x = pendulum.x.copy()

                action_selection += 1
                if action_selection > ACTION_SELECTION_THRESHOLD:
                    u = policy(x, Q_network)
                    action_selection = 1

                next_x, cost = pendulum.step(u)

                cost_to_go += discount * cost
                discount *= DISCOUNT_FACTOR

                final = True if i == EPISODE_LENGTH - 1 else False
                buffer.add_transition(x=x, u=u, cost=cost, next_x=next_x,
                                      is_final=final)

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
                    is_final_batch = [b[4] for b in batch]

                    _loss = update(x_batch, u_batch, cost_batch, x_next_batch,
                                   is_final_batch, Q_network_target, Q_network,
                                   critic_optimizer, DISCOUNT_FACTOR, nu)
                    gradients_update = 1
                    data[e]['loss'].append(_loss)  # Takes the float value

                target_update += 1
                if target_update > TARGET_UPDATE_THRESHOLD:
                    Q_network_target.set_weights(Q_network.get_weights())
                    target_update = 1

                pbar.update(1)

            pbar.close()

        print("Cost to go:", cost_to_go)
        save_network += 1
        average_cost_to_go += cost_to_go
        if save_network > SAVE_NETWORK_THRESHOLD:
            average_cost_to_go /= SAVE_NETWORK_THRESHOLD
            if average_cost_to_go < best_average_cost_to_go:
                print("Saving network with {} average cost to go"
                      .format(average_cost_to_go))
                Q_network.save_weights(BEST_WEIGHTS_FILE_PATH)
                best_average_cost_to_go = average_cost_to_go
            average_cost_to_go = 0
            save_network = 1

        if e == EPISODES - 1:
            Q_network.save_weights(TRAINED_WEIGHTS_FILE_PATH)

        proportion = e / EPISODES
        EPSILON = max(EPSILON_MIN, np.exp(EPSILON_DECAY * proportion))
        print("Epsilon", EPSILON)
        policy.epsilon = EPSILON
        print("")

    # test last trained networks
    test_episodes = 3

    Q_network.load_weights(TRAINED_WEIGHTS_FILE_PATH)
    print("Testing of the network after the last episode"
          " from {} random starting positions".format(test_episodes))
    for i in range(3):
        test_network(pendulum, Q_network, policy,
                     record_namefile='Last_episode_%d' % i,
                     record_folder=MOVIE_DIR)

    '''
    print("Testing of the network after the last"
          " episode {} times from down position".format(test_episodes))
    for i in range(3):
        # a bit of randomness to the down position
        # q is in [pi-random,pi+random]
        # no randomness to velocity
        q = np.pi + np.random.rand(nq)*(0.2-(-0.2))+(-0.2)
        v = np.zeros(nq)
        state = np.hstack([q, v])
        test_network(pendulum, Q_network, policy, state,
                     record_namefile='Last_episode_down_%d' % i,
                     record_folder=MOVIE_DIR)

    Q_network.load_weights(BEST_WEIGHTS_FILE_PATH)
    print("Testing of the best network from"
          " {} random starting positions".format(test_episodes))
    for i in range(3):
        test_network(pendulum, Q_network, policy,
                     record_namefile='Last_episode_random_%d' % i,
                     record_folder=MOVIE_DIR)

    print("Testing of the best network"
          " {} times from down position".format(test_episodes))
    for i in range(3):
        # a bit of randomness to the down position
        # q is in [pi-random,pi+random]
        # no randomness to velocity
        q = np.pi + np.random.rand(nq) * (0.2 - (-0.2)) + (-0.2)
        v = np.zeros(nq)
        state = np.hstack([q, v])
        test_network(pendulum, Q_network, policy, state,
                     record_namefile='Best_network_%d' % i,
                     record_folder=MOVIE_DIR)
    '''
