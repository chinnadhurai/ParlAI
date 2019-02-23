# Perutbation operations
import numpy as np
np.random.seed(seed=300)


class Perturb(object):
    def __init__(self, opt):
        self.opt = opt
        self.splitter = "\n"

    def perturb(self, act):
        if self.opt['perturb'] == 'None':
            return act
        turns = self._get_turns(act)
        if 'random' in self.opt['perturb']:
            perturb_op = np.random.choice([self.swap, self.drop, self.repeat])
            turns = perturb_op(turns)
        elif "drop" in self.opt['perturb']:
            turns = self.drop(turns)
        elif "swap" in self.opt['perturb']:
            turns = self.swap(turns)
        elif "repeat" in self.opt['perturb']:
            turns = self.repeat(turns)
        else:
            assert "Invalid perturb mode : {}. Valid : random, drop, swap, repeat".format(self.opt['perturb'])
        
        self._update_act(turns, act)
        return act

    def swap(self, turns):
        if "first" in self.opt['perturb']:
            pos = [0,1]
        elif "last" in self.opt['perturb']:
            last_id = len(turns) - 1
            pos = [last_id-1, last_id]
        else:
            pos = np.random.randint(len(turns), size=2)
        tmp = turns[pos[0]]
        turns[pos[0]] = turns[pos[1]]
        turns[pos[1]] = tmp
        return turns
        
    def drop(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = np.random.randint(len(turns))
        return [turn for idx, turn in enumerate(turns) if idx != pos]

    def repeat(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = np.random.randint(len(turns))
        return turns[:pos] + [turns[pos]] + turns[pos:]

    def _get_turns(self, act):
        return act['text'].split('\n')

    def _update_act(self, turns, act):
        act['text'] = self.splitter.join(turns)
