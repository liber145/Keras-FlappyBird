import sys
sys.path.append("./game")
import wrapped_flappy_bird as game
import numpy as np


class FlappyBirdEnv(object):
    def __init__(self):
        self.env = game.GameState()
        self._dim_observation = (288, 512, 3)
        self._dim_action = 2

    def reset(self):
        self.env.__init__()
        obs, rew, done = self.env.frame_step(np.array([1, 0]))
        return obs

    def step(self, act):
        a = [0, 0]
        a[act] = 1
        obs, rew, done = self.env.frame_step(a)
        return obs, rew, done, dict()

    def sample_action(self):
        return np.random.randint(self.dim_action)

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action


if __name__ == "__main__":
    env = FlappyBirdEnv()
    s = env.reset()
    for i in range(100):
        s, r, d, info = env.step(env.sample_action())
        print(f"s: {s.shape}, r: {r}, d: {d}, info: {info}")
