import gymnasium as gym

# Starting a new environment
env = gym.make("CartPole-v1", render_mode="human") # Render_mode controls how the enviroment show to me, human so it render in real time so good for watching

# Prints that shows what each space contain
########################################################################################################
########################################################################################################
# Discrete action space (button presses)
env = gym.make("CartPole-v1")
print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}")  # Box with 4 values
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation
########################################################################################################
########################################################################################################

# Always reset the environment first
observation, info = env.reset() # Gives me first observation with list of what agent can "see"

# prints a list of [cart position, cart velocity, pole angle, pole angular velocity]
print(f"Starting Observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
  action = env.action_space.sample() # random action for now, real agent will be smarter
  observation, reward, terminated, truncated, info = env.step(action) # make the action to environment and see what happened

  total_reward += reward
  episode_over = terminated or truncated # When terminated or truncated the while loop stop

print(f"Episode finished! Total reward: {total_reward}")
env.close()



