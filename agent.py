import torch

#### PyTorch Actor Critic Model
class Agent(torch.nn.Module):
    def __init__(self, in_size=10, hidden=64, out=1):
        super(Agent, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out),
            torch.nn.Softmax(dim=0)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        self.actor[-2].weight.data.fill_(0.0)

    def act(self, state):
        state = state.view(-1)
        #batch = state.shape[0]

        action = self.actor(state.float().view(-1))
        values = self.critic(state.float().view(-1))
        return action, values
