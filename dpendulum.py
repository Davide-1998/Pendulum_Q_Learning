from pendulum import Pendulum
import numpy as np


class DPendulum:
    ''' Discrete Pendulum environment. Joint angle, velocity and torque are
        discretized
        with the specified steps. Joint velocity and torque are saturated.
        Guassian noise can be added in the dynamics.
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''

    def __init__(self, nu=11, dt=0.2, ndt=1, noise_stddev=0, joints=1):
        self.pendulum = Pendulum(joints, noise_stddev)
        self.pendulum.DT = dt
        self.pendulum.NDT = ndt

        self.nu = nu        # Number of discretization steps for joint torque
        self.uMax = self.pendulum.umax    # Max torque (u in [-umax,umax])
        self.dt = dt        # time step
        self.DU = 2*self.uMax/self.nu  # discretization resolution for joint torque

    @property
    def nq(self): return self.pendulum.nq
    ''' Size of the v vector '''
    @property
    def nv(self): return self.pendulum.nv
    ''' Size of the x vector '''
    @property
    def nx(self): return self.nq+self.nv

    @property
    def goal(self): return np.zeros(self.nx)

    # Continuous to discrete
    def c2du(self, u):
        u = np.clip(u, -self.uMax+1e-3, self.uMax-1e-3)
        u_d = np.floor((u+self.uMax)/self.DU).astype(int)
        return u_d

    # Discrete to continuous
    def d2cu(self, iu):
        iu = np.clip(iu, 0, self.nu-1) - (self.nu-1)/2
        return iu*self.DU

    def controls(self):
        controls = list(range(self.nu))
        controls = np.array(controls)
        controls = np.tile(controls, (self.nq, 1))
        return controls

    def reset(self, x=None):
        self.x = self.pendulum.reset(x)
        return self.x.copy()

    def step(self, iu):
        u = self.d2cu(iu)
        self.x, cost = self.pendulum.step(u)
        return self.x.copy(), cost

    def render(self):
        self.pendulum.render()

    def dynamics(self, ix, iu):
        x = ix.copy()
        u = self.d2cu(iu)
        xc, cost = self.pendulum.dynamics(x, u)
        return xc.copy(), cost
