from gym.core import Wrapper
from baselines import logger
class RB_Monitor(Wrapper):

    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.reward_components = []

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        if info.get('rews') is not None:
            self.reward_components.append(info.get('rews'))
            del info['rews']
            if done:
                rewinfo = self.reward_components[0]
                for step in self.reward_components[:-1]:
                    for k, v in step.items():
                        if k in rewinfo:
                             rewinfo[k] += v
                        else:
                            rewinfo[k] = v
                self.reward_components = []
                info['episode'] = dict(info['episode'], **rewinfo)
        return (ob, rew, done, info)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

