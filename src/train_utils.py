import torch
import wandb
from agent_torch_openai import Teacher

def train_epoch_naive(env, gamma, episode, actor, actor_optimizer, critic, critic_optimizer, visualise):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    log_probs = []
    values = []
    rewards = []
    entropy_term = 0

    done = False
    total_reward = 0

    while not done:
        # Get action probabilities from the actor network
        probs = actor(state)
        # Create a categorical distribution over the list of probabilities of actions
        dist = torch.distributions.Categorical(probs)
        # Sample an action using the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # Save log prob, value estimates, and reward
        value = critic(state)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        entropy_term += entropy.mean()

        state = next_state
        total_reward += reward

        if done:
            Qval = 0
            values = torch.cat(values)
            # Compute the Q values
            Qvals = []
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + gamma * Qval
                Qvals.insert(0, Qval)
            Qvals = torch.cat(Qvals)

            # Convert lists to tensors
            log_probs = torch.stack(log_probs)
            advantage = Qvals - values.squeeze()

            # Calculate actor and critic loss
            actor_loss = (-log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            total_loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_term

            # Backpropagate and update weights
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

            print(f"Episode {episode}, Total Reward: {total_reward}")
            wandb.log({f"loss_overall": total_loss})
            wandb.log({f"reward_overall": total_reward})
            wandb.log({f"loss_{env.task}": total_loss})
            wandb.log({f"reward_{env.task}": total_reward})
            return
        
def train_epoch_Round_Robin(env, gamma, episode, actor, actor_optimizer, critic, critic_optimizer, visualise):
    tasks = env.task_names
    print("<<< RR >>>")
    for task in tasks:
        env.reset_task(task_name=task)
        state = env.reset()
        # print(task, env.task)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        log_probs = []
        values = []
        rewards = []
        entropy_term = 0

        done = False
        total_reward = 0

        while not done:
            # Get action probabilities from the actor network
            probs = actor(state)
            # Create a categorical distribution over the list of probabilities of actions
            dist = torch.distributions.Categorical(probs)
            # Sample an action using the distribution
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action.item())
            if visualise:
                next_state = next_state["image"]
            else:
                next_state = next_state["features"]
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            # Save log prob, value estimates, and reward
            value = critic(state)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            entropy_term += entropy.mean()

            state = next_state
            total_reward += reward

            if done:
                Qval = 0
                values = torch.cat(values)
                # Compute the Q values
                Qvals = []
                for t in reversed(range(len(rewards))):
                    Qval = rewards[t] + gamma * Qval
                    Qvals.insert(0, Qval)
                Qvals = torch.cat(Qvals)

                # Convert lists to tensors
                log_probs = torch.stack(log_probs)
                advantage = Qvals - values.squeeze()

                # Calculate actor and critic loss
                actor_loss = (-log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_term

                # Backpropagate and update weights
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

                print(f"Episode {episode}, Task {env.task}, Total Reward: {total_reward}")
                wandb.log({f"loss_{task}": total_loss})
                wandb.log({f"reward_{task}": total_reward})
    return

def train_epoch_openai_teacher(env, gamma, episode, actor, actor_optimizer, critic, critic_optimizer, visualise):
    teacher = Teacher(env.task_names, gamma=gamma)
