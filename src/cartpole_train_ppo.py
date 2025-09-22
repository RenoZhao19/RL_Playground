# What is PPO?
# Proximal Policy Optimization(think proximal = stay close)
""" 
!!!In RL there are two big families:
  1. Value-based (like Q-learning): learnes how good is each action?
  2. Policy-based (like PPO): directly learn the policy(probability of choosing actions)

Why are we using PPO instead of other policy based method:
- like vinila policy gradient, update could be too big which breaks the policy
- learning could be unstable
While PPO is taking small step when updating the policy, slowly finding the best policy

How does PPO work?
1. Collect data by running policy in the environment
  - save the state, action, reward, next state

2. Calculate advantage
  - Advantage > 0 good action which encourage it
  - viceversa when adv. < 0 discourage it

3. Update policy (with clipping)
  - old policy thinks action A have prob. of 0.2
  - new policy thinks 0.9
  - clip the change, bc PPO think it's too big of a jump
  - prevent agent from destroying itself by overcorrecting

4. Repeat the process then update again

IMPORTANT THING: "ADVANTAGE"!!!!
  - ADVANTAGE is like a report card per action
  - If the action has positive advantage then encourage it EX: In one simulation pushing right at
    state X helps balance much longer this has a positive advantage! So we want more like this in 
    X state!
  - We need advantae because if we only use raw reward we would push every action ina good 
    episode equally - even the random un-helpful one
  - ADVANTAGE FOCUS ON CREDIT ASSIGNMENT: which actions truly matter!

All in all Proximal Policy Optimization is an on policy actor-critic algorithm, it learns:
- a policy that picks action
- a value function that estimates how good a state is

*** Notes that PPO doesn't know "balance the pole" explicitly it only knows choose action that 
increase long term reward!

*** Policy = Probability distribution 
    pi(a|s) = Probability of taking action a given a state s
    so for each state PPO maintain a probability distribution over action
    we have pi(a|s) = [P(left|s),P(right|s)]

Important Key Words
1. MlpPolicy = multi-layer perceptron
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO

# helper func to run my trained agent for several episodes and average the total rewards
from stable_baselines3.common.evaluation import evaluate_policy 

# Creating a place to record model and notes of my run001, and some os command
OUTDIR = "results/cartpole/run001"
os.makedirs(OUTDIR, exist_ok = True) # make directory at root, and even if the file exist won't crash

env = gym.make("CartPole-v1")
# MlpPolicy = multi-layer perceptron(a feed forward neural net)
# for stable_baselines3 there's also CnnPolicy for image inputs
model = PPO("MlpPolicy", env, verbose = 1, seed = 67) # picked verbose = 1 to print reward time steps(read friendly), 0 is silent and 2 is very detailed debug info
model.learn(total_timesteps = 100000) # 100000 times step(environment interaction), 100000 left or right training throughout many episodes
model.save(f"{OUTDIR}/model") # saving our model for later use

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes = 20)
with open(f"{OUTDIR}/notes.md", "w") as notes:
  notes.write("# CartPole PPO - Run 001\n")
  notes.write("Time steps = 100,000, seed = 67\n")
  notes.write(f"Evaluation(20 Episodes): mean = {mean_reward:.2f} +/- {std_reward:.2f}\n")
print(f"Evaluation with 20 episodes: mean = {mean_reward:.2f} +/- {std_reward:.2f}\n")

env.close()
