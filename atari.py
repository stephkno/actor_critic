import gym
import torch
import datetime
import random
import sys
from matplotlib import pyplot as plt
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
    state = torch.tensor(state)/255.0
    return state
def learn(episode):
    R = 0.0
    log_probs = []
    value_loss = []
    returns = []

    #state value calculation & preprocessing
    for i,(_,_,reward) in enumerate(reversed(episode)):
        R = reward + (GAMMA * R)
        returns.insert(0,R)

    returns = torch.tensor(returns)
    mean = returns.mean()
    std = returns.std() if returns.std() > 0 else 1
    returns = (returns - mean)/std

    log_prob, values, _ = zip(*episode)
    episode = list(zip(log_prob, values, returns))

    #calculate policy error
    for i, (log_prob, state_value, returns) in enumerate(random.sample(episode,k=len(episode))):
        advantage = returns - state_value
        log_probs.append(-log_prob * advantage)
        value_loss.append(0.5*(state_value - returns.unsqueeze(0)).pow(2))

    #update parameters
    print("Updating parameters")
    optimizer.zero_grad()
    loss = torch.cat((torch.cat(log_probs), torch.cat(value_loss))).mean()
    loss.backward()
    print("Loss: {}".format(loss.item()))
    optimizer.step()
    memory.reset()
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
GAMMA = 0.9
EPSILON = 1.0
rate = 0.001
UPDATE_INTERVAL = 1
TARGET_UPDATE_INTERVAL = 10
running_scores = []
plot = True

env_name = "CartPole-v0"
env = gym.make(env_name)

#create objects
memory = Memory()
agent = Agent(env.action_space.n)
coach = Coach()

optimizer = torch.optim.Adam(params=agent.parameters(), lr=rate)
print(agent)

#training loop
while True:

    memory, score, steps = coach.run_episode(agent, env, memory, preprocess, test)
    total_score += score
    total_steps += steps

    print("[Episode {} Score:{}]".format(episode, score))
    running_scores.append(score)

    if not test and episode % UPDATE_INTERVAL == 0:
        learn(memory.episode)

        if score > highest:
            save_model(agent)
            highest = score
        if plot:
            plt.plot(range(episode), running_scores)
            plt.draw()
            plt.pause(0.0000001)
        #display episode score
        print("[Episode {}|Step {} Score:{} High:{} Time:{}]".format(episode, total_steps, total_score, highest, datetime.datetime.now()))
        total_score = 0

    episode += 1

env.close()
