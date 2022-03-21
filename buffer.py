import random
from tqdm import tqdm
import numpy as np


class ExperienceReplay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = list()

    def size(self):
        return len(self.memory)

    def max_size(self):
        return self.max_size

    def add_transition(self, **kwargs):
        state = kwargs["x"]
        control = kwargs["u"]
        cost = kwargs["cost"]
        next_state = kwargs["next_x"]
        is_final_state = kwargs["is_final"]

        if self.size() > self.max_size:
            self.memory.pop(0)

        t = (state, control, cost, next_state, is_final_state)
        self.memory.append(t)

    def sample(self, batch_size=1):
        n = min(batch_size, self.size())
        return random.sample(self.memory, n)

    def fill(self, num_ep, len_ep, robot, policy, critique, selection_th):
        action_selection = 0
        with tqdm(total=num_ep) as pbar:
            pbar.set_description("Filling experience replay")
            filling_episodes = int(num_ep/len_ep)
            for i in range(filling_episodes):
                robot.reset()
                u = robot.c2du(np.zeros(robot.nq))

                for j in range(len_ep):
                    x = robot.x.copy()

                    action_selection += 1
                    if action_selection > selection_th:
                        u = policy(x, critique)
                        action_selection = 1

                    next_x, cost = robot.step(u)
                    final = True if i == int(num_ep/len_ep) - 1 else False
                    self.add_transition(x=x, u=u, cost=cost, next_x=next_x,
                                        is_final=final)
                    pbar.update(1)
            pbar.close()
