# Perutbation operations


class Perturb(object):
    def __init__(self, opt, i2t, t2i):
        self.opt = opt
        self.i2t = i2t
        self.t2i = t2i
        

    def perturb(act):
        turns = self._act_to_turns(act)
        self.swap(turns)
        self._update_act(turns, act)
        return act

    def swap(turns, positions=None):
        pass

    def drop(turns, pos=None):
        pass

    def repeat(turns, pos=None):
        pass

    def _get_turns(act):
        pass

    def _update_act(turns, act):
        pass
