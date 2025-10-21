import argparse
import gym
import crafter
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
import wandb
import sys
import numpy as np

# Optionally import WandbCallback from sb3_contrib if using an older version
from wandb.integration.sb3 import WandbCallback

MAX_STEPS=1000
TASKS = None

# achievements:
# collect_coal
# collect_diamond
# collect_drink
# collect_iron
# collect_sapling
# collect_stone
# collect_wood
# defeat_skeleton
# defeat_zombie
# eat_cow
# eat_plant
# make_iron_pickaxe
# make_iron_sword
# make_stone_pickaxe
# make_stone_sword
# make_wood_pickaxe
# make_wood_sword
# place_furnace
# place_plant
# place_stone
# place_table
# wake_up

# Custom Callback to log reward and loss to wandb
class WandbRewardLossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbRewardLossCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log reward and loss to wandb
        # print(self.locals)
        if self.locals.get('reward') is not None:
            # Logging the reward for the current step
            wandb.log({"reward": self.locals['reward']})
        
        # Check if 'loss' is available in the locals and log it
        if 'loss' in self.locals:
            wandb.log({"loss": self.locals['loss']})
        
        return True

def main(args):
    # Define the make_env function for creating multiple environments
    def make_env(rank):
        def _init():
            # Create the Crafter environment
            env = gym.make('CrafterReward-v1', tasks=TASKS, invincible=True)  # Or use CrafterNoReward-v1 depending on your preference
            env = crafter.Recorder(
                env, f'logs/crafter',
                save_stats=False,
                save_video=False,
                save_episode=False,
            )
            # env = TaskSamplingEnv(env, TASKS, if_random_reset_task=IF_RANDOM_RESET_TASK)
            env = Monitor(env)
            return env
        return _init

    num_envs = args.n_envs  # Number of parallel environments

    # Create the vectorized environment
    # envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    envs = gym.make('CrafterReward-v1', tasks=TASKS, invincible=True)
    envs = Monitor(envs)

    # Initialize wandb for logging
    wandb.init(
        project=f"{args.wandb_project}-{args.algo}",
        config=vars(args),
        sync_tensorboard=True,
        name=args.wandb_run_name,
    )

    # Set up evaluation callback to periodically evaluate and save the model
    eval_callback = EvalCallback(
        envs,  # Use the training environment for evaluation
        best_model_save_path=args.best_model_save_path,
        log_path=args.log_path,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    # Create a custom callback to log reward and loss
    reward_loss_callback = WandbRewardLossCallback()
    callback = CallbackList([eval_callback, WandbCallback(verbose=1), reward_loss_callback])

    # Dictionary to map algorithm names to constructors
    ALGOS = {
        'dqn': DQN,
        'ppo': PPO,
        'a2c': A2C,
        'sac': SAC,
        'td3': TD3,
    }

    # Ensure the selected algorithm is valid
    assert args.algo.lower() in ALGOS.keys(), f"Invalid algorithm. Choose from {list(ALGOS.keys())}"

    # Select the algorithm class
    AlgoClass = ALGOS[args.algo.lower()]

    # Create policy keyword arguments
    policy_kwargs = {}
    if args.net_arch:
        # Example for setting network architecture
        net_arch = [int(x) for x in args.net_arch.split(',')]
        policy_kwargs['net_arch'] = net_arch

    # Prepare model hyperparameters
    model_kwargs = {
        'env': envs,
        # 'n_steps': MAX_STEPS,
        # 'tensorboard_log': args.tensorboard_log,
        'learning_rate': args.learning_rate,
        # 'exploration_initial_eps': 1.0,  # Start with 100% exploration, dqn
        # 'exploration_final_eps': 0.1,    # Reduce to 10% exploration, dqn
        # 'exploration_fraction': 0.9,     # dqn
        'gamma': 0.99,
    }

    # Some algorithms have specific hyperparameters
    if args.algo.lower() in ['dqn', 'sac', 'td3']:
        model_kwargs['buffer_size'] = args.buffer_size

    # Instantiate the agent
    model = AlgoClass(
        args.policy,
        **model_kwargs,
        policy_kwargs=policy_kwargs
    )

    log_path = "./logs/sb3_log/"
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=1
    )

    # Save the trained agent
    model.save(args.model_save_path)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    # Argument parsing and setup (unchanged)
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stable Baselines3 and wandb")

    # Algorithm and Environment arguments
    parser.add_argument('--algo', type=str, default='ppo', help='RL Algorithm: dqn, ppo, a2c, sac, td3')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy type: MlpPolicy, CnnPolicy, etc.')

    # Hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=1e8, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=500000, help='Replay buffer size (for off-policy algorithms)')
    parser.add_argument('--net_arch', type=str, default='512, 512, 256, 256, 128', help='Network architecture (comma-separated numbers)')
    parser.add_argument('--learning_tasks', type=str, default='All', help='Learning Tasks: All, Single, Multi')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=5e4, help='Evaluation frequency in timesteps')
    parser.add_argument('--best_model_save_path', type=str, default='./logs/best_model', help='Path to save the best model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Directory to save evaluation logs')

    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='acrl-test-sb-crafter', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')

    # Logging and saving paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/', help='TensorBoard log directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    args = parser.parse_args()

    main(args)