import argparse
import gym
import wandb
import yaml
import numpy as np
import sys
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from wandb.integration.sb3 import WandbCallback

# Import your custom environment
sys.path.append('../../')
sys.path.append('../../craft_env')
import craft_env
from craft_env.env_gym import TaskSamplingEnv, TASKS

SEPERATE = True
VISUALIZE = False
IF_RANDOM_RESET_TASK = True
MAX_STEPS=1000
ACCUMLATE_REWARD = True

# Load the curriculum from a YAML file
def load_curriculum(config_path):
    with open(config_path, 'r') as f:
        curriculum = yaml.safe_load(f)
    return curriculum['curriculum']

# Custom Callback to track curriculum and change tasks based on steps
class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum = curriculum
        self.current_step = 0
        self.current_tasks = self.curriculum[0]['tasks']  # Start with the first tasks

    def _on_step(self) -> bool:
        tasks_cache = self.current_tasks
        total_steps = self.num_timesteps
        for stage in self.curriculum:
            if total_steps >= stage['steps']:
                tasks_cache = stage['tasks']
            if total_steps < stage['steps']:
                break
        if tasks_cache != self.current_tasks:
            self.current_tasks = tasks_cache
            print(f"Changing tasks to: {self.current_tasks}")
            self._update_env_tasks()
        return True

    def _update_env_tasks(self):
        if isinstance(self.current_tasks, list):
            self.model.get_env().envs[0].env.reset_task(self.current_tasks)  # Dynamically update tasks
            # print(self.model.get_env().envs[0].env.env.env._legal_tasks) # dummyvec/orderenforcing/craftenv
        else:
            self.model.get_env().envs[0].env.env._legal_tasks = None

def main(args):
    # Load curriculum from YAML file
    curriculum = load_curriculum(args.curriculum_file)

    # Define the make_env function for creating multiple environments
    def make_env(rank=None):
        def _init():
            env = gym.make(id='CraftEnv-v0', visualise=VISUALIZE, max_steps=MAX_STEPS, accumulate_reward=ACCUMLATE_REWARD)
            # env = TaskSamplingEnv(env, TASKS, if_random_reset_task=True)
            env = Monitor(env)
            return env
        return _init

    num_envs = args.n_envs  # Number of parallel environments

    # Initialize wandb for logging
    wandb.init(
        project=f"{args.wandb_project}-{args.algo}",
        config=vars(args),
        sync_tensorboard=True,
        name=args.wandb_run_name,
    )

    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = DummyVecEnv([make_env()])
    # env = Monitor(env)

    # Set up evaluation callback to periodically evaluate and save the model
    eval_callback = EvalCallback(
        env,  # Use the training environment for evaluation
        best_model_save_path=args.best_model_save_path,
        log_path=args.log_path,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    # Create the curriculum callback
    curriculum_callback = CurriculumCallback(curriculum)
    callback = CallbackList([eval_callback, WandbCallback(verbose=1), curriculum_callback])

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
        net_arch = [int(x) for x in args.net_arch.split(',')]
        policy_kwargs['net_arch'] = net_arch

    # Prepare model hyperparameters
    model_kwargs = {
        'env': env,
        'verbose': 1,
        'learning_rate': args.learning_rate,
        'exploration_initial_eps': 1.0,  # Start with 100% exploration, dqn
        'exploration_final_eps': 0.1,    # Reduce to 10% exploration, dqn
        'exploration_fraction': 0.9,     # dqn
        'gamma': 0.99,
    }

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
        callback=callback
    )

    # Save the trained agent
    model.save(args.model_save_path)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stable Baselines3 and wandb")

    # Algorithm and Environment arguments
    parser.add_argument('--algo', type=str, default='dqn', help='RL Algorithm: dqn, ppo, a2c, sac, td3')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy type: MlpPolicy, CnnPolicy, etc.')

    # Hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=1e8, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=500000, help='Replay buffer size (for off-policy algorithms)')
    parser.add_argument('--net_arch', type=str, default='512, 512, 256, 256, 128', help='Network architecture (comma-separated numbers)')
    parser.add_argument('--learning_tasks', type=str, default='All', help='Learning Tasks: All, Single, Multi')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1e5, help='Evaluation frequency in timesteps')
    parser.add_argument('--best_model_save_path', type=str, default='./logs/best_model', help='Path to save the best model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Directory to save evaluation logs')

    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='acrl-test-sb-accumulate-curriculum', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')

    # Logging and saving paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/', help='TensorBoard log directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')

    # Curriculum learning
    parser.add_argument('--curriculum_file', type=str, default='curriculum.yaml', help='Path to the curriculum YAML file')

    args = parser.parse_args()

    main(args)