import gymnasium as gym
import numpy as np
from ppo import Agent
from utils import plot_learning_curve
from torchvision.transforms import transforms


def process_state_image(state):
    transform = transforms.Grayscale()
    monochrome_tensor = transform(state.permute(2, 0, 1)).squeeze()
    # monochrome_tensor /= 255
    return monochrome_tensor

if __name__ == '__main__':
    action1 = 0
    action4 = 0
    env = gym.make('CarRacing-v2', domain_randomize=False, render_mode='human', continuous=False)
    N = 20
    batch_size = 64
    n_epochs = 1
    alpha = 0.0003
    print(env.observation_space.shape)
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=[3, 96, 96])
    n_games = 200

    figure_file = 'plots/carracingplot.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _  = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            # print(action)
            observation_, reward, done1, done2, info = env.step(action)

            if action1 == 4 or action4 == 4:
                reward -= 2
            else:
                reward += 0.5
            if action == 1:
                action1 += 1
                action4 = 0
            elif action == 4:
                action4 += 1
                action1 = 0
            else:
                action1 = 0
                action4 = 0
            if action == 0:
                reward -= 1.5
            done = done1 or done2
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
        agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
