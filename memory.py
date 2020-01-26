from collections import namedtuple
Transition = namedtuple('Transition', ('log_prob', 'state_value', 'reward'))

#replay memory for episodes
class Memory():
    def __init__(self):
        super(Memory, self).__init__()
        self.episode = []
        self.reset()

    def push(self, *args):
        self.episode.append(Transition(*args))
        self.length += 1

    def sample(self):
        return self.targets

    def reset(self):
        self.episode = []
        self.length = 0
