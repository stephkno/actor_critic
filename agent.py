import torch

#### PyTorch Actor Critic Model
class Agent(torch.nn.Module):
    def __init__(self, n):
        super(Agent, self).__init__()

        self.actor = torch.nn.GRU(4, 64, 1)
        self.critic = torch.nn.GRU(4, 64, 1)

        self.actor_head = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(64, n, bias=False),
            torch.nn.Softmax(dim=0)
        )
        self.critic_head = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1, bias=False),
        )
        self.actor_head[-2].weight.data.fill_(0.0)

    def reset(self):
        self.actor_hidden = torch.zeros(1,1,64)
        self.critic_hidden = torch.zeros(1,1,64)

    def act(self, state):
        state = state.view(-1)
        #batch = state.shape[0]

        action, self.actor_hidden = self.actor(state.float().unsqueeze(0).unsqueeze(0), self.actor_hidden)
        values, self.critic_hidden = self.critic(state.float().unsqueeze(0).unsqueeze(0), self.critic_hidden)

        action = self.actor_head(action.view(-1))
        values = self.critic_head(values.view(-1))
        return action, values
