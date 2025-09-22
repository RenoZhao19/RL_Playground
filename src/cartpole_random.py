import gymnasium as gym

# Outcome I've Observed: only survice around 15-20 step with this random smaple
# Next I will try train with PPO
def main():
  env = gym.make("CartPole-v1", render_mode = "human") # version is v1 
  obs, info = env.reset()
  total_points = 0
  episode_end = False

  while not episode_end:
    # EACH push are fixed with force of +or- 10 Newton(N) to the cart
    action = env.action_space.sample() # Still we have a random action from action_space(0 or 1)
    obs, reward, terminated, truncated, info = env.step(action) # Terminated is we Lost/Win, truncated is times up!
    total_points += reward
    if terminated or truncated:
      episode_end = True
  print(f"This random policy for agents survived {int(total_points)} steps.")
  env.close()

if __name__ == "__main__":
  main()
  
