import argparse
import sys
import gym
import wandb
import numpy as np

# Import your custom environment
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../craft_env')

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from gym import Wrapper
# from plot import plot_success_rates

# Optionally import WandbCallback from sb3_contrib if using an older version
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
# Set train frequency to a certain number of steps
train_freq = TrainFreq(1000, unit=TrainFrequencyUnit.STEP)


import craft_env
from craft_env.env_gym import TaskSamplingEnv, N_TASKS, TASKS, TASKS_SIMPLE_MAKE, TASKS_SIMPLE_GET, TASKS_HARD

SEPERATE = True
VISUALIZE = False
IF_RANDOM_RESET_TASK = True
MAX_STEPS=1000
ACCUMLATE_REWARD = True
TASKS = TASKS_HARD

# Define the learning rate schedule
def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

def main(args):
    # Define the make_env function for creating multiple environments
    def make_env(rank):
        def _init():
            env = gym.make(id='CraftEnv-v0', visualise=VISUALIZE, max_steps=MAX_STEPS, accumulate_reward=ACCUMLATE_REWARD)
            if not ACCUMLATE_REWARD:
                env = TaskSamplingEnv(env, TASKS, if_random_reset_task=IF_RANDOM_RESET_TASK)
            env = Monitor(env)
            return env
        return _init

    num_envs = args.n_envs  # Number of parallel environments

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Initialize wandb for logging
    wandb.init(
        project=f"{args.wandb_project}-{args.algo}",
        config=vars(args),
        sync_tensorboard=True,
        name=args.wandb_run_name,
    )

    # Set up evaluation callback to periodically evaluate and save the model
    eval_callback = EvalCallback(
        env,  # Use the training environment for evaluation
        best_model_save_path=args.best_model_save_path,
        log_path=args.log_path,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    def make_eval_env_fn(task_name):
        env = gym.make(id='CraftEnv-v0', visualise=False, max_steps=MAX_STEPS, accumulate_reward=ACCUMLATE_REWARD)
        if not ACCUMLATE_REWARD:
            env = TaskSamplingEnv(env, [task_name], if_random_reset_task=IF_RANDOM_RESET_TASK)
        env = Monitor(env)
        return env
    
    # Create the task success rate callback with parallel evaluation
    task_success_callback = TaskSuccessRateCallback(
        tasks=TASKS,
        make_env_fn=make_eval_env_fn,
        eval_freq=args.eval_freq,
        n_eval_episodes=100
    )

    # Combine callbacks: evaluation, wandb logging, and task success[] rates
    callback = CallbackList([eval_callback, WandbCallback()])
    # callback = CallbackList([eval_callback, WandbCallback(), task_success_callback])

    # Verify the environment follows the Gym interface
    obs = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation shape:", obs[0].shape)  # Access the first environment's observation
    # check_env(env) # Commented out because check_env does not support VecEnvs

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
        'env': env,
        'verbose': 1,
        # 'n_steps': MAX_STEPS,
        # 'batch_size': 2048,
        # 'exploration_initial_eps': 1.0,  # Start with 100% exploration, dqn
        # 'exploration_final_eps': 0.1,    # Reduce to 10% exploration, dqn
        # 'exploration_fraction': 0.9,     # dqn
        'tensorboard_log': args.tensorboard_log,
        # 'learning_rate': linear_schedule(args.learning_rate),
        'learning_rate': args.learning_rate,
        'gamma': 0.99,
        'train_freq': 5,
        # 'train_freq': train_freq       # ddpg
        # 'n_steps': 10,
        # 'use_rms_prop': True
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

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback
    )

    # Save the trained agent
    model.save(args.model_save_path)

    # Plot success rates after training
    # plot_success_rates(
    #     success_rates=task_success_callback.success_rates,
    #     eval_steps=task_success_callback.eval_steps,
    #     tasks=TASKS,
    #     save_path="success_rates.png"
    # )

    # Finish the wandb run
    wandb.finish()

# Custom callback to track success rates per task with parallel evaluation
class TaskSuccessRateCallback(BaseCallback):
    """
    Custom callback for evaluating success rates across multiple tasks in parallel.
    """
    def __init__(self, tasks, make_env_fn, eval_freq=10000, n_eval_episodes=10, verbose=0):
        super(TaskSuccessRateCallback, self).__init__(verbose)
        self.tasks = tasks
        self.make_env_fn = make_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_rates = {task: [] for task in tasks}
        self.eval_steps = []
        self.task_vec_envs = {}
        
        # Initialize a vectorized environment for each task
        for task in tasks:
            # Create a list of environment constructors for parallel episodes
            env_fns = [lambda task=task: self.make_env_fn(task) for _ in range(n_eval_episodes)]
            vec_env = DummyVecEnv(env_fns)
            self.task_vec_envs[task] = vec_env

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            wandb.log({'training_step': self.num_timesteps})
            for task, vec_env in self.task_vec_envs.items():
                # Reset all environments for this task
                obs = vec_env.reset()
                dones = [False] * vec_env.num_envs
                episode_rewards = np.zeros(vec_env.num_envs)
                
                while not all(dones):
                    # Predict actions for all environments in parallel
                    actions, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = vec_env.step(actions)
                    episode_rewards += rewards
                    # Reset environments that have finished their episodes
                    for i, done in enumerate(dones):
                        if done:
                            obs[i] = vec_env.env_method('reset', indices=i)[0]
                
                # Determine successes based on total rewards
                successes = (episode_rewards > 0).sum()
                success_rate = successes / self.n_eval_episodes
                self.success_rates[task].append(success_rate)
                
                # Log success rate to wandb
                wandb.log({f'success_rate_{task}': success_rate, 'eval_step': self.num_timesteps})
            
            self.eval_steps.append(self.num_timesteps)
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stable Baselines3 and wandb")

    # Algorithm and Environment arguments
    parser.add_argument('--algo', type=str, default='ppo', help='RL Algorithm: dqn, ppo, a2c, sac, td3')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy type: MlpPolicy, CnnPolicy, etc.')

    # Hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=1e8, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=5000000, help='Replay buffer size (for off-policy algorithms)')
    parser.add_argument('--net_arch', type=str, default='512, 512, 256, 256, 128', help='Network architecture (comma-separated numbers)')
    parser.add_argument('--learning_tasks', type=str, default='All', help='Learning Tasks: All, Single, Multi')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1e5, help='Evaluation frequency in timesteps')
    parser.add_argument('--best_model_save_path', type=str, default='./logs/best_model', help='Path to save the best model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Directory to save evaluation logs')

    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='acrl-test-sb-accumulate', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')

    # Logging and saving paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/', help='TensorBoard log directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    args = parser.parse_args()

    main(args)
