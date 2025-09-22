import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode = "human")
model = PPO.load("results/cartpole/run001/model")

observation , info = env.reset(seed = 123) # Now we test our model on a random seed, but same random

total = 0
episode_end = False

# Note we can get max env step via: MAX_STEPS = env.spec.max_episode_steps

# I can do this while loop becasue when cartpole reached a step of 500 it will terminate
while not episode_end: # For some other RL env it doesn't terminate so we need to set a step cap
  action, _ = model.predict(observation, deterministic = True) # _ means ignore the Stable baseline internnal info
  observation, rewards, terminated, truncated, info =  env.step(action)
  total += rewards
  if terminated or truncated: episode_end = True

print(f"This agents survived {int(total)} steps")

# Always remember to close the environment(otherwise might have memory leak or multiple ghost window)
env.close()