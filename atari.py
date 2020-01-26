import gym
import torch
import sys
from collections import namedtuple
import random
import datetime

env = gym.make("Breakout-ramNoFrameskip-v0")

if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    test = True
else:
    test = False

torch.manual_seed(0)

def preprocess(state):
    #state extraction for Pong RAM
    #state = torch.tensor(state[[0x31, 0x36, 0x38, 0x3A, 0x3C]])/255.0
    state = torch.tensor(state)/255.0
    return state

#### PyTorch Actor Critic Model
class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(64, env.action_space.n, bias=False),
            torch.nn.Softmax(dim=0)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1, bias=True),
        )
        self.actor[-2].weight.data.fill_(0.0)

    def act(self, state):
        #batch = state.shape[0]
        action = self.actor(state.float()).view(-1)
        values = self.critic(state.float()).view(-1)
        return action, values

#tuple for episode transitions
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

    def learn(self):
        R = 0.0
        log_probs = []
        value_loss = []
        returns = []

        #state value calculation & preprocessing
        for i,(_,_,reward) in enumerate(reversed(self.episode)):
            if reward < 0:
                R = -1.0
            elif reward > 0:
                R = 1.0
            else:
                R = reward + (GAMMA * R)
            returns.insert(0,R)

        returns = torch.tensor(returns)
        mean = returns.mean()
        std = returns.std() if returns.std() > 0 else 1
        returns = (returns - mean)/std

        log_prob, values, _ = zip(*self.episode)
        self.episode = list(zip(log_prob, values, returns))

        #calculate policy error
        for i, (log_prob, state_value, returns) in enumerate(random.sample(self.episode,k=len(self.episode))):
            advantage = returns - state_value
            log_probs.append(-log_prob * advantage)
            value_loss.append((state_value - returns.unsqueeze(0)).pow(2))

        #update parameters
        print("Updating parameters")
        optimizer.zero_grad()
        loss = torch.cat((torch.cat(log_probs), torch.cat(value_loss))).mean()
        loss.backward()
        print("Loss: {}".format(loss.item()))
        optimizer.step()
        self.reset()

#loading/saving checkpoint for testing
def load_agent():
    print("Loading agent")
    agent.load_state_dict(torch.load("./checkpoint.pth"))
def save_model(model):
    print(" ~!  ---- Saving model ---- !~")
    torch.save(model.state_dict(), './checkpoint.pth')

#hyperparameters
step = 0
highest = -9999 #high score
total_score = 0.0
total_steps = 0
batch_size = 10000
lives = 0
init_frameskip = 1
epoch = 0
epochs = 1000
episode = 1
init_action = 1
GAMMA = 0.9
EPSILON = 0.2
rate = 0.0002
UPDATE_INTERVAL = 1

#create objects
memory = Memory()
agent = Agent()
optimizer = torch.optim.Adam(params=agent.parameters(), lr=rate)
print(agent)

#training loop
while True:
    done = False

    score = 0.0
    confidence = 1.0

    #reset game
    state = env.reset()
    state = preprocess(state)

    env._max_episode_steps = 5000
    render = test

    if test:
        load_agent()

    #initial framekskip to get lives
    for i in range(init_frameskip):
        _, _, _, info = env.step(init_action)
        if 'ale.lives' in info:
            lives = info["ale.lives"]

    #episode loop
    while not done:
        action_probs, state_value = agent.act(state)
        dist = torch.distributions.Categorical(action_probs)
        r = random.random()

        if r < EPSILON and not test:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)
        next_state, reward, done, info = env.step(int(action))
        next_state = preprocess(next_state)
	
        #check if lost atari life
        if 'ale.lives' in info and info["ale.lives"] < lives:
            lives = info["ale.lives"]
            reward = -1.0

        score += reward
        total_score += reward
        step += 1
        total_steps += 1

        if done:
            reward = -1.0

        #render env for observing agent
        if render:
            env.render()

        #prime next state
        state = next_state
        memory.push(dist.log_prob(action), state_value, reward)


    print("[Episode {} Score:{}]".format(episode, score))

    if not test and episode % UPDATE_INTERVAL == 0:
        memory.learn()

        if score > highest:
            save_model(agent)
            highest = score

        #display episode score
        print("[Episode {}|Step {} Score:{} High:{} Time:{}]".format(episode, total_steps, total_score, highest, datetime.datetime.now()))
        total_score = 0

    episode += 1

env.close()
