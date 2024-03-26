from SAC import SAC
import gymnasium as gym
from ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("CarRacing-v2",render_mode='')
    agent = SAC()
    # agent.load()
    replay_buffer = ReplayBuffer()
    maxReturn = -float('inf')
    return_list = []
    for episode in range(100000):
        state , _ = env.reset()
        
        episode_return = 0
        for step in range(60):
            env.step([0,0,0])
        stop = 0
        for step in range(5000):
            action = agent.take_action(state)
            next_state, reward , done , _ , _ = env.step(action)
            episode_return += reward
            if agent.judge_out_of_route(next_state):
                reward -= 10
                done = True
            if reward < 0 and step>20:
                stop -= 1
            else:
                stop = 0
            if stop < -20:
                done=True
            replay_buffer.add(state,action,reward,next_state,done)
            state = next_state
            if replay_buffer.size()>100:
                states , actions , rewards , next_states , dones = replay_buffer.sample(64)
                agent.update(states,actions,rewards,next_states,dones)
            if done:
                break
        print(f"\repisond:{episode} return:{episode_return}",end='')
        return_list.append(episode_return)
        # agent.save()
        plt.plot(return_list,label='reward')
        plt.legend()
        plt.title("Rewards")
        plt.xlabel("episodes")
        plt.ylabel("Reward")
        plt.savefig("figure.png")
        plt.close()