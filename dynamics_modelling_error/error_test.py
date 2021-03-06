import numpy as np
from dpendulum import DPendulum


def test(robot, initial_state=None, episode_length=256):
    robot.reset(initial_state)
    for _ in range(episode_length):
        # always use the control corresponding to 0 torque
        control = robot.c2du(np.zeros(robot.nq))
        robot.step(control)
        robot.render()


if __name__ == "__main__":
    pendulum = DPendulum()

    print("starting to test with almost down position")
    for i in np.linspace(1., 0.1, 10):
        b = i
        print(f"Testing with b = {b}")
        pendulum.changeB(b)
        # q = pi is the down position, we also add a bit of randomness
        q = np.pi + np.random.rand(pendulum.nq) * (0.3 - (-0.3)) + (-0.3)
        # v is the velocity of the joint
        v = np.zeros(pendulum.nq)
        state = np.hstack([q, v])
        test(pendulum, state)

    print("starting to test with almost horizontal position")
    for i in np.linspace(1., 0.1, 10):
        b = i
        print(f"Testing with b = {b}")
        pendulum.changeB(b)
        # q = pi/4 is the horizontal position, we also add a bit of randomness
        q = np.pi/4 + np.random.rand(pendulum.nq) * (0.3 - (-0.3)) + (-0.3)
        # v is the velocity of the joint
        v = np.zeros(pendulum.nq)
        state = np.hstack([q, v])
        test(pendulum, state)

