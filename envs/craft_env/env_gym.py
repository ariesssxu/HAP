import gym
from gym import spaces
import numpy as np
import collections
import yaml
import warnings
from gym import Wrapper
# Your existing imports
import craft
import env, env_new_reward
from misc import util

# get local path
import os
local_path = os.path.dirname(os.path.realpath(__file__))

N_TASKS = 24
TASKS = ['get[gem]', 'get[gold]', 'get[grass]', 'get[iron]', 'get[rock]', 'get[wood]', \
    'make[arrow]', 'make[axe]', 'make[bed]', 'make[bench]', 'make[bow]', 'make[bridge]', \
        'make[bundle]', 'make[cloth]', 'make[flag]', 'make[goldarrow]', 'make[hammer]', \
            'make[knife]', 'make[ladder]', 'make[plank]', 'make[rope]', 'make[shears]', 'make[slingshot]', 'make[stick]']
TASKS_SIMPLE_GET = ["get[wood]", "get[grass]", "get[iron]", "get[rock]"] # simple get, OK
TASKS_SIMPLE_MAKE = ["make[plank]", "make[stick]", "make[cloth]", "make[rope]"] # simple make, OK
TASKS_HARD = ["make[bed]", "make[axe]", "make[shears]", "get[gold]"]

# Define the Task namedtuple
Task = collections.namedtuple("Task", ["goal", "steps"])

# Custom environment wrapper to sample tasks
class TaskSamplingEnv(Wrapper):
    # ['get[gem]', 'get[gold]', 'get[grass]', 'get[iron]', 'get[rock]', 'get[wood]', 'make[arrow]', 'make[axe]', 'make[bed]', 'make[bench]', 'make[bow]', 'make[bridge]', 'make[bundle]', 'make[cloth]', 'make[flag]', 'make[goldarrow]', 'make[hammer]', 'make[knife]', 'make[ladder]', 'make[plank]', 'make[rope]', 'make[shears]', 'make[slingshot]', 'make[stick]']
    def __init__(self, env, tasks, if_random_reset_task):
        super(TaskSamplingEnv, self).__init__(env)
        self.tasks = tasks
        # print("Available tasks:", self.tasks)
        self.current_task_index = None
        self.if_random_reset_task = if_random_reset_task

    def reset(self, task_id=None):
        if task_id is not None:
            self.current_task_index = task_id
        elif self.if_random_reset_task:
            self.current_task_index = np.random.choice(len(self.tasks))
        else:
            self.current_task_index = 0  # Default to the first task
        # Set the environment to the current task
        # Assuming the environment has a method to set tasks
        self.env.reset_task(self.tasks[self.current_task_index])

        observation = self.env.reset()
        return observation

    def step(self, action):
        return self.env.step(action)
    
class CraftEnv(gym.Env):
    """Gym environment for the Craft game."""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 recipes_path=f"{local_path}/resources/recipes.yaml",
                 hints_path=f"{local_path}/resources/hints.yaml",
                 max_steps=100,
                 seed=0,
                 visualise=False,
                 reuse_environments=False,
                 accumulate_reward=False,
            ):
        super(CraftEnv, self).__init__()

        # Initialize your environment factory
        self.factory = EnvironmentFactory(
            recipes_path=recipes_path,
            hints_path=hints_path,
            max_steps=max_steps,
            seed=seed,
            visualise=visualise,
            reuse_environments=reuse_environments
        )

        # Sample an initial environment
        self.env = self.factory.sample_environment()
        self.visualise = visualise
        self.task_name = None
        self.max_steps = max_steps
        self.task_one_hot = np.zeros(N_TASKS)
        self._legal_tasks = None
        # Define action and observation spaces
        # You'll need to specify these based on your environment
        self.action_space = self._define_action_space()
        self.accumulate_reward = accumulate_reward
        self.observation_space = self._define_observation_space()

    # def _define_action_space(self):
    #     # Example: Discrete action space with N possible actions
    #     # Replace N with the number of actions in your environment
    #     N = self.env.action_space_size
    #     return spaces.Discrete(N)

    # def _define_observation_space(self):
    #     # Example: Observation space with shape and bounds
    #     # Adjust low, high, shape according to your observations
    #     obs_shape = self.env.observation_shape
    #     obs_low = np.full(obs_shape, -np.inf)
    #     obs_high = np.full(obs_shape, np.inf)
    #     return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    
    def _define_action_space(self):
        # Suppose there are 5 discrete actions
        return spaces.Discrete(5)
        # return spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def _define_observation_space(self):
        # Suppose observations are images with shape (64, 64, 3)
        if self.env._visualise:
            return spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        else:
            # (1586,)
            # return spaces.MultiBinary(4640+N_TASKS)
            if self.accumulate_reward:
                return spaces.Box(low=0, high=1, shape=(176,), dtype=np.float32)
            else:
                return spaces.Box(low=0, high=1, shape=(176+N_TASKS,), dtype=np.float32)

    def step(self, action):
        # Step through the environment using the provided action
        if isinstance(action, np.ndarray):
            action = np.clip(action, self.action_space.low, self.action_space.high)
            action = np.argmax(action)
            # print(action)
        reward, done, obs = self.env.step(action)
        info = {}
        obs = obs["image"] if self.visualise else obs["features"]
        if not self.accumulate_reward and not self.visualise:
            assert np.sum(self.task_one_hot) == 1
            obs = np.concatenate([obs, self.task_one_hot], axis=0)
        return obs, reward.astype(float) - 0.5/self.max_steps, done, info

    def seed(self, seed=None):
        # Seed the environment to ensure reproducibility
        # self.env.seed(seed)
        return
    
    def reset_task(self, task_name=None):
        self.task_name = task_name
        self.task_one_hot = np.zeros(N_TASKS)
        self.task_one_hot[self.task_names.index(task_name)] = 1
        self.env = self.factory.sample_environment(task_name=task_name)

    def reset(self):
        # Reset the environment and return the initial observation
        self.env = self.factory.sample_environment(task_name=self.task_name, accumulate_reward=self.accumulate_reward)
        print("aaa", self._legal_tasks)
        # assert 0
        self.env._legal_tasks = self._legal_tasks
        obs = self.env.reset()
        obs = obs["image"] if self.visualise else obs["features"]
        if self.accumulate_reward:
            return obs
        elif self.visualise:
            return obs
        else:
            return np.concatenate([obs, self.task_one_hot], axis=0)

    def render(self, mode='human'):
        # Render the environment if visualization is enabled
        if self.env._visualise:
            return self.env.render_matplotlib()
            # return self.env.render_frame()
            # return self.env.render_curses()

    def close(self):
        # Close the environment and clean up resources
        self.env.close()

    @property
    def task_names(self):
        return self.factory.task_names
    
    @property
    def task(self):
        return self.env.task_name

class EnvironmentFactory(object):
    """Factory instantiating Craft environments compatible with Gym."""

    def __init__(self,
                 recipes_path,
                 hints_path,
                 max_steps=100,
                 seed=0,
                 visualise=False,
                 reuse_environments=False):
        self.subtask_index = util.Index()
        self.task_index = util.Index()
        self._max_steps = max_steps
        self._visualise = visualise
        self._reuse_environments = reuse_environments

        # Per task, we reuse the same environment, with same layouts.
        # Should generate easier tasks where agents can overfit.
        if self._reuse_environments:
            self._env_cache = {}

        # Create World
        self.world = craft.CraftWorld(recipes_path, seed)

        # Load the tasks with sub-steps (== hints)
        with open(hints_path) as hints_f:
            self.hints = yaml.load(hints_f, Loader=yaml.FullLoader)

        # Setup all possible tasks
        self._init_tasks()

    def _init_tasks(self):
        """Build the list of tasks and subtasks."""
        # Organize task and subtask indices
        self.tasks_by_subtask = collections.defaultdict(list)
        self.tasks = {}
        for hint_key, hint in self.hints.items():
            # Parse the hint key and steps
            goal = util.parse_fexp(hint_key)
            goal = (self.subtask_index.index(goal[0]),
                    self.world.cookbook.index[goal[1]])
            steps = tuple(self.subtask_index.index(s) for s in hint)
            task = Task(goal, steps)
            for subtask in steps:
                self.tasks_by_subtask[subtask].append(task)

            self.tasks[hint_key] = task
            self.task_index.index(task)

        self.task_names = sorted(self.tasks.keys())

        if self._reuse_environments:
            # Preallocate environments to handle random seed consistency
            for task_name in self.task_names:
                self.sample_environment(task_name)

    def _create_environment(self, task_name, accumulate_reward=False):
        # Get the task
        task = self.tasks[task_name]
        goal_arg = task.goal[1]

        # Sample a world (== scenario)
        scenario = self.world.sample_scenario_with_goal(goal_arg)

        # Wrap it into an environment and return
        if accumulate_reward:
            return env_new_reward.CraftLab(
                scenario,
                task_name,
                task,
                max_steps=self._max_steps,
                visualise=self._visualise)
        else:
            return env.CraftLab(
                scenario,
                task_name,
                task,
                max_steps=self._max_steps,
                visualise=self._visualise)

    def sample_environment(self, task_name=None, accumulate_reward=False):
        if task_name is None:
            task_name = np.random.choice(self.task_names)

        if self._reuse_environments:
            return self._env_cache.setdefault(task_name,
                                              self._create_environment(task_name))
        else:
            return self._create_environment(task_name, accumulate_reward)

if __name__ == "__main__":
    # Replace with the paths to your recipes and hints files
    recipes_path = "resources/recipes.yaml"
    hints_path = "resources/hints.yaml"

    env = CraftEnv(recipes_path, hints_path)
    obs = env.reset()
    done = False

    while not done:
        # Replace with your agent's action selection logic
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()