import gymnasium as gym
import envs
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import tianshou as ts
from tianshou.utils.space_info import SpaceInfo
from tianshou.policy.base import BasePolicy
import pickle 

def main() -> None:
    task = "envs/PacMan-v0"
    render_mode = "symbol"
    train_flag = "dqn-symbol"
    lr, epoch, batch_size = 1e-3, 100, 64
    train_num, test_num = 128, 64
    gamma, n_step, target_freq = 0.99, 3, 320
    buffer_size = 100000
    eps_train, eps_test = 0.15, 0.05
    step_per_epoch, step_per_collect = 10000, 64

    logger = ts.utils.TensorboardLogger(SummaryWriter(f"log/{train_flag}"))  # TensorBoard is supported!
    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html
    log_path = f"log/{train_flag}"

    # You can also try SubprocVectorEnv, which will use parallelization
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, render_mode=render_mode) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, render_mode=render_mode) for _ in range(test_num)])

    from tianshou.utils.net.common import Net

    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    env = gym.make(task, render_mode=render_mode)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy: ts.policy.DQNPolicy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        action_space=env.action_space,
        estimation_step=n_step,
        target_update_freq=target_freq,
    )
    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_num),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector(
        policy,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False
    
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path
    
    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=stop_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    print(f"Finished training in {result.timing.total_time} seconds")

    # watch performance
    policy.set_eps(eps_test)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1000, render=1 / 35)

    torch.save(policy.state_dict(), f"{log_path}/dqn.pth")


if __name__ == "__main__":
    main()