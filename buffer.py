import random


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

        if self.size() > self.max_size:
            self.memory.pop(0)

        t = (state, control, cost, next_state)
        self.memory.append(t)

    def sample(self, batch_size=1):
        n = min(batch_size, self.size())
        return random.sample(self.memory, n)
