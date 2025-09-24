# RL Playground

Weekend reinforcement learning project towards AI/robotics

## Week 1 = CarPole with PPO(Stable_Baseline3)
- **Learning Concept**: agent, environment, state, action, reward, policy, episode
- **Environments**: `CartPole-v1` (Gymnasium)
- **Algorithm Used to Train Model**: Proximal Policy Optimization(PPO)

### Results (Run 001)
- Time steps = 100,000, seed = 67
- Evaluation(20 Episodes): mean = **500.00 +/- 0.00** (target >= 450)
- Artifact: `results/cartpole/run001/model.zip`, `notes.md`

### How to Run?
python3 -m venv .venv # Creating the virtual environment

source .venv/bin/activate # Activating the virtual environment

pip install -r requirements.txt

python3 src/cartpole_random.py

python3 src/cartpole_train_ppo.py

python3 cartpole_evaluation.py # Shows the render window run on trained ppo model
