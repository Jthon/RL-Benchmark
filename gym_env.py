import gym
import config as cfg
import cv2
class gym_env:
    def __init__(self):
        self.env=gym.make(cfg.params["level_name"])
        self.env.seed(cfg.params["seed"])
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        #Enviroment Config
        print("Level="+cfg.params["level_name"])
        print("ob_space={}".format(self.observation_dim))
        print("ac_space={}".format(self.action_dim))
    def render(self):
        rgb=self.env.render(mode="rgb_array")
        cv2.imshow("render",rgb)
        cv2.waitKey(1)
        return rgb
    def reset(self):
        return self.env.reset()
    def step(self,action):
        next_ob, rew, done, _=self.env.step(action)
        return next_ob,rew,done