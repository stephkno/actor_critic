import gym
import torch
import datetime
import random
import sys
from matplotlib import pyplot as plt
from collections import namedtuple
from agent import Agent
from memory import Memory
from coach import Coach

if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    test = True
else:
    test = False

torch.manual_seed(0)

def preprocess(state):
    #state extraction for Pong RAM
    #state = torch.tensor(state[[0x31, 0x36, 0x38, 0x3A, 0x3C]])/255.0
    state = torch.tensor(state)
    return state
def learn(agent, episode):

    optimizer = torch.optim.Adam(params=agent.parameters(), lr=rate)
    R = 0.0
    log_probs = []
    value_loss = []
    returns = []

    #state value calculation & preprocessing
    for i,(_,_,reward,_) in enumerate(reversed(episode)):
        R = reward + (GAMMA * R)
        returns.insert(0,R)

    returns = torch.tensor(returns)
    mean = returns.mean()
    std = returns.std() if returns.std() > 0 else 1
    returns = (returns - mean)/std

    log_prob, values, _, entropy = zip(*episode)
    episode = list(zip(log_prob, values, returns))
    #entropy = torch.tensor(entropy).mean()

    #calculate policy error
    for i, (log_prob, state_value, returns) in enumerate(random.sample(episode,k=len(episode))):
        advantage = returns - state_value
        log_probs.append(-log_prob * advantage)
        value_loss.append((state_value - returns.unsqueeze(0)).pow(2))

    #update parameters
    optimizer.zero_grad()
    loss = torch.cat(log_probs).mean() + torch.cat(value_loss).mean()
    loss.backward()
    optimizer.step()
    memory.reset()

    return agent
#loading/saving checkpoint for testing
def load_agent():
    print("Loading agent")
    agent.load_state_dict(torch.load("./checkpoint.pth"))
def save_model(model):
    print(" ~!  ---- Saving model ---- !~")
    torch.save(model.state_dict(), './checkpoint.pth')

#hyperparameters
step = 0
highest = -9999
total_score = 0.0
total_steps = 0
batch_size = 10000
lives = 0
init_frameskip = 1
epoch = 0
epochs = 1000
episode = 1
init_action = 1
GAMMA = 0.99
EPSILON = 1.0
rate = 0.0005
UPDATE_INTERVAL = 10
PLOT_INTERVAL = 1
plot = True
running_scores = []

env_name = "CartPole-v0"
env = gym.make(env_name)

#create objects
memory = Memory(namedtuple('Transition', ('log_prob', 'state_value', 'reward', 'entropy')))
def reward_shaping(reward, done):
    return 0 if not done else 1

agent = Agent(env.action_space.n)
coach = Coach(reward_shaping=reward_shaping)

print(agent)

if test:
    load_agent()

#training loop
while True:
    memory, score, steps = coach.run_episode(agent, env, memory, preprocess, test)
    total_score += score
    total_steps += steps

    #print("[Episode {} Score:{}]".format(episode, score))
    if not test and episode % UPDATE_INTERVAL == 0:
        agent = learn(agent, memory.episode)

        if episode % PLOT_INTERVAL == 0:
            if plot:
                plt.plot(range(len(running_scores)), running_scores)
                plt.draw()
                plt.pause(0.0000001)

            #display episode score
            print("[Episode {}|Step {} Score:{} High:{} Time:{}]".format(episode, total_steps, total_score, highest, datetime.datetime.now()))

            if total_score > highest:
                save_model(agent)
                highest = total_score
            running_scores.append(total_score)
            total_score = 0

    episode += 1

env.close()
