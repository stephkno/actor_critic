import torch
import random

class Coach():
    def __init__(self):
        super(Coach, self).__init__()
        self.init_frameskip = 0
        self.init_action = 0
        self.EPSILON = 1.0

    def run_episode(self, agent, env, memory, preprocess, test):
        done = False

        score = 0.0
        step = 0

        # reset game
        agent.reset()
        state = env.reset()
        state = preprocess(state)

        env._max_episode_steps = 5000
        render = test

        # initial framekskip to get lives
        for i in range(self.init_frameskip):
            _, _, _, info = env.step(self.init_action)
            if 'ale.lives' in info:
                lives = info["ale.lives"]

        # episode loop
        while not done:
            action_probs, state_value = agent.act(state)

            dist = torch.distributions.Categorical(action_probs)
            t_dist = torch.distributions.Categorical(action_probs)

            r = random.random()

            if r < self.EPSILON:
                action = dist.sample()
            else:
                action = torch.argmax(action_probs)

            next_state, reward, done, info = env.step(int(action))
            next_state = preprocess(next_state)

            # check if lost atari life
            if 'ale.lives' in info and info["ale.lives"] < lives:
                lives = info["ale.lives"]
                reward = -1.0
                done = True

            score += reward
            step += 1

            if done:
                reward = -1.0

            reward = 0 if not done else -1

            if render:
                env.render()

            # prime next state
            state = next_state
            memory.push(t_dist.log_prob(action), state_value, reward)

        return memory, score, step
