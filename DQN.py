import os.path

import tensorflow as tf
# from tensorflow.keras import layers
# for some reason in this build the import above gives the error:
# cannot find reference 'keras' in '__init__.py'
from keras import layers
import numpy as np
from orca.orca import init
import json
from manipulator.dpendulum import DPendulum
from buffer import ExperienceReplay
from policy import EpsilonGreedy
from time import time
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
        batch_Q_values = np.reshape(Q_values, (-1, 1))  # 64x1
        # print('Q_Value Shape', batch_Q_values.shape)
        # print('States Shape', states_batch.shape)  # 64x4

        # print('Next States Shape', next_states_batch.shape)  # 64x4

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
                 makeMovie=False,
                 record_namefile='', record_folder=os.getcwd()):
    robot.reset(initial_state)
    episode_cost = 0
    episode_cost_history = []
    discount_factor = 1

    if makeMovie:
        robot.pendulum.record_pendulum(custom_namefile=record_namefile,
                                       movie_dir=record_folder)

    for _ in range(episode_length):
        state = robot.x.copy()
        control = pi.optimal(state, Q)
        next_state, step_cost = robot.step(control)
        episode_cost += discount_factor * step_cost
        episode_cost_history.append(episode_cost)
        discount_factor *= DISCOUNT_FACTOR
        robot.render()

    if makeMovie:
        robot.pendulum.end_record(record_namefile, record_folder)
    return episode_cost, episode_cost_history


def training(n_joints, q_levels, num_episodes, len_episodes, exp_replay_size,
             batch_size, no_op_th, discount_factor, epsilon_start=1,
             epsilon_max=1,
             epsilon_min=0.01, epsilon_decay=0.01, target_update_th=15,
             gradient_descent_th=4, action_selection_th=1, save_network_th=15,
             learning_rate=1e-4,
             best_weights_dir=os.getcwd(), trained_weights_dir=os.getcwd()):

    pendulum = DPendulum(joints=n_joints, nu=q_levels)
    target_update = 0
    gradients_update = 0
    action_selection = 0
    save_network = 0

    # Experience replay initialization
    buffer = ExperienceReplay(exp_replay_size)

    # Policy initialization
    policy = EpsilonGreedy(epsilon_start, pendulum.controls())

    # Create critic and target NNs
    nx = pendulum.nx
    nu = pendulum.nu
    nq = pendulum.nq
    Q_network = get_critic(nx, nu ** nq)
    Q_network_target = get_critic(nx, nu ** nq)

    # Set initial weights of targets equal to those of actor and critic
    Q_network_target.set_weights(Q_network.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

    data = {'training_loss': [], 'training_time': [], 'epsilon': []}

    # filling the experience replay buffer
    buffer.fill(no_op_th, len_episodes,
                pendulum, policy, Q_network, 2)

    average_cost_to_go = 0
    best_average_cost_to_go = np.Inf

    # Training

    for e in range(num_episodes):
        x = pendulum.reset()
        u = pendulum.c2du(np.zeros(pendulum.nq))
        cost_to_go = 0
        discount = 1

        avg_loss = []
        with tqdm(total=len_episodes) as pbar:
            pbar.set_description('Episode %d' % (e + 1))
            start_time = time()
            for i in range(len_episodes):

                x = pendulum.x.copy()

                action_selection += 1
                if action_selection > action_selection_th:
                    u = policy(x, Q_network)
                    action_selection = 1

                next_x, cost = pendulum.step(u)

                cost_to_go += discount * cost
                discount *= discount_factor

                final = True if i == len_episodes - 1 else False
                buffer.add_transition(x=x, u=u, cost=cost, next_x=next_x,
                                      is_final=final)

                gradients_update += 1
                if gradients_update > gradient_descent_th:
                    batch = buffer.sample(batch_size)
                    x_batch = np.array([b[0] for b in batch])
                    u_batch = np.array([b[1] for b in batch])
                    cost_batch = np.array([b[2]
                                          for b in batch]).reshape((-1, 1))
                    x_next_batch = np.array([b[3] for b in batch])
                    is_final_batch = [b[4] for b in batch]

                    _loss = update(x_batch, u_batch, cost_batch, x_next_batch,
                                   is_final_batch, Q_network_target, Q_network,
                                   critic_optimizer, discount_factor, nu)
                    _loss = _loss.tolist()
                    gradients_update = 1
                    if isinstance(_loss, float):
                        avg_loss.append(_loss)
                    else:
                        # Average loss
                        avg_loss.append(sum(_loss)/len(_loss))

                target_update += 1
                if target_update > target_update_th:
                    Q_network_target.set_weights(Q_network.get_weights())
                    target_update = 1

                pbar.update(1)

            pbar.close()
        data['training_loss'].append(sum(avg_loss)/len(avg_loss))
        data['training_time'].append(time() - start_time)

        print("Cost to go:", cost_to_go)
        save_network += 1
        average_cost_to_go += cost_to_go
        if save_network > save_network_th:
            average_cost_to_go /= save_network_th
            if average_cost_to_go < best_average_cost_to_go:
                print("Saving network with {} average cost to go"
                      .format(average_cost_to_go))
                Q_network.save_weights(best_weights_dir)
                best_average_cost_to_go = average_cost_to_go
            average_cost_to_go = 0
            save_network = 1

        if e == num_episodes - 1:
            Q_network.save_weights(trained_weights_dir)

        # proportion = e / num_episodes
        data['epsilon'].append(epsilon_start)  # Save to data
        epsilon_start = max(epsilon_min, np.exp(-epsilon_decay * e))
        print("Epsilon", epsilon_start)
        policy.epsilon = epsilon_start

    training_environment = {'robot': pendulum, 'Q_network': Q_network,
                            'Q_network_target': Q_network_target,
                            'policy': policy}
    return training_environment, data


def test(test_eps, Q_network, policy, pendulum,
         last_ep_weights_dir=os.getcwd(), best_net_weights_dir=os.getcwd(),
         movie_dir=os.getcwd()):

    test_episodes = test_eps
    data = {}
    nq = pendulum.nq

    movie_descriptor = ''
    for key, val in {'ep': test_episodes, 'res': pendulum.nu,
                     'joints': pendulum.pendulum.model.njoints}.items():
        movie_descriptor += '{}_key'.format(val)

    Q_network.load_weights(last_ep_weights_dir)
    print("Testing of the network after the last episode"
          " from {} random starting positions".format(test_episodes))
    data['loss_last_ep_random_pos'] = {}
    for i in range(test_episodes):
        cost = test_network(pendulum, Q_network, policy,
                            record_namefile='Last_episode_random_'
                            + movie_descriptor,
                            record_folder=movie_dir)
        data['loss_last_ep_random_pos'][i] = cost[1]

    print("Testing of the network after the last"
          " episode {} times from down position".format(test_episodes))
    data['loss_last_ep_down_pos'] = {}
    for i in range(test_episodes):
        # a bit of randomness to the down position
        # q is in [pi-random,pi+random]
        # no randomness to velocity
        q = np.pi + np.random.rand(nq)*(0.2-(-0.2))+(-0.2)
        v = np.zeros(nq)
        state = np.hstack([q, v])
        cost = test_network(pendulum, Q_network, policy, state,
                            record_namefile='Last_episode_down_'
                            + movie_descriptor,
                            record_folder=movie_dir)
        data['loss_last_ep_down_pos'][i] = cost[1]

    Q_network.load_weights(best_net_weights_dir)
    print("Testing of the best network from"
          " {} random starting positions".format(test_episodes))
    data['loss_best_net_random_pos'] = {}
    for i in range(test_episodes):
        cost = test_network(pendulum, Q_network, policy,
                            record_namefile='Best_network_random_'
                            + movie_descriptor,
                            record_folder=movie_dir)
        data['loss_best_net_random_pos'][i] = cost[1]

    print("Testing of the best network"
          " {} times from down position".format(test_episodes))
    data['loss_best_net_down_pos'] = {}
    for i in range(test_episodes):
        # a bit of randomness to the down position
        # q is in [pi-random,pi+random]
        # no randomness to velocity
        q = np.pi + np.random.rand(nq) * (0.2 - (-0.2)) + (-0.2)
        v = np.zeros(nq)
        state = np.hstack([q, v])
        cost = test_network(pendulum, Q_network, policy, state,
                            record_namefile='Best_network_down_'
                            + movie_descriptor,
                            record_folder=movie_dir)
        data['loss_best_net_down_pos'][i] = cost[1]
    return data


if __name__ == "__main__":

    #####################################
    # Custom global parameters          #
    #####################################

    MOVIE_DIR = os.getcwd() + os.sep + 'Movie'
    JSON_DIR = os.getcwd() + os.sep + 'Collected_Data'

    BEST_WEIGHTS_FILE_PATH = os.path.abspath("weights/best_network_weights.h5")
    TRAINED_WEIGHTS_FILE_PATH = os.path.abspath(
        "weights/last_network_weights.h5")

    #####################################
    # Hyper-parameters                  #
    #####################################

    NUMBER_OF_JOINTS = 2
    # the number of quantization levels for controls should be an odd number
    QUANTIZATION_LEVELS = 15

    EPISODES = 100
    EPISODE_LENGTH = 2 ** 8

    EXPERIENCE_REPLAY_SIZE = 2 ** 16
    BATCH_SIZE = 2 ** 6
    NO_OP_THRESHOLD = 2 ** 14
    DISCOUNT_FACTOR = 0.99

    EPSILON = 1
    EPSILON_MAX = 1
    EPSILON_MIN = 0.01
    EPSILON_DECAY = -1 * (np.log(EPSILON_MIN) / (0.75 * EPISODES))
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

    ###########################################################################

    # Setup and training of the system
    training_env, data = training(NUMBER_OF_JOINTS, QUANTIZATION_LEVELS,
                                  EPISODES, EPISODE_LENGTH,
                                  EXPERIENCE_REPLAY_SIZE,
                                  BATCH_SIZE, NO_OP_THRESHOLD, DISCOUNT_FACTOR,
                                  EPSILON, EPSILON_MAX, EPSILON_MIN,
                                  EPSILON_DECAY, TARGET_UPDATE_THRESHOLD,
                                  GRADIENT_DESCENT_THRESHOLD,
                                  ACTION_SELECTION_THRESHOLD,
                                  SAVE_NETWORK_THRESHOLD, LEARNING_RATE,
                                  BEST_WEIGHTS_FILE_PATH,
                                  TRAINED_WEIGHTS_FILE_PATH)

    # Test of system ##########################################################
    test_results = test(3, training_env['Q_network'], training_env['policy'],
                        training_env['robot'], TRAINED_WEIGHTS_FILE_PATH,
                        BEST_WEIGHTS_FILE_PATH, MOVIE_DIR)

    # Concatenate results #####################################################
    data.update(test_results)

    # Save results locally ####################################################
    if not os.path.isdir(JSON_DIR):
        os.mkdir(JSON_DIR, 0o777)

    fileName = 'Results_{}_joints_{}_ep_{}_len_{}_res.json'.format(
                                                            NUMBER_OF_JOINTS,
                                                            EPISODES,
                                                            EPISODE_LENGTH,
                                                            QUANTIZATION_LEVELS
                                                            )
    filePath = JSON_DIR + os.sep + fileName
    io_out = open(filePath, 'w')
    json.dump(data, io_out, indent=4)
    io_out.close()
