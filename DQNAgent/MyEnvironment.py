import random
from gym import Env


class MyEnviroment(Env):
    def __init__(self, x, y):
        self.train_X = x
        self.train_Y = y.tolist()
        self.myset = set(self.train_Y)
        self.current_index = self._sample_index()
        self.action_space = 1000

    def reset(self):
        obs, r, done, _ = self.step(-1)
        return obs

    def update_data(self, x, y):
        self.train_X = x
        self.train_Y = y.tolist()
        self.myset = set(self.train_Y)

    def step(self, action):
        done = False
        if action == -1:
            info = {'action': -1}
            _c_index = self.current_index
            self.current_index = self._sample_index()
            res = self.train_X[_c_index]
            res = res.unsqueeze(0)
            return res, 0, done, info
        r = self.reward(action)
        if r == 1:
            done = True

        self.current_index = self._sample_index()
        res = self.train_X[self.current_index]
        res = res.unsqueeze(0)
        info = {'action': action}
        return res, r, done, info

    def reward(self, action):
        c = self.train_Y[self.current_index]
        return 100 if c == action else -1
        # return 1 if c == action else -(abs(c - action)) / 1000

    def sample_actions(self):
        return self.myset[random.randint(0, len(self.myset))]
        # return self.train_Y[random.randint(0, len(self.train_Y) - 1)]
        # return random.randint(0, self.action_space)

    def _sample_index(self):
        return random.randint(0, len(self.train_Y) - 1)
