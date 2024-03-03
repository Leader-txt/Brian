from collections import deque
import random

class ReplayBuffer():
    def __init__(self,maxlen=100000):
        self.buffer = deque(maxlen=maxlen)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append([state,action,[reward],next_state,[done]])
    def sample(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        states , actions , rewards , next_states , dones = zip(*batch)
        return states , actions , rewards , next_states , dones
    def size(self):
        return len(self.buffer)