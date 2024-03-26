import Net_NoCnn as Net
import torch
import torch.optim as optim
import torch.nn as nn
from apex import amp
import numpy as np

class SAC():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.Policy = Net.Policy().to(self.device)
        self.Critic1 = Net.Critic().to(self.device)
        self.Critic2 = Net.Critic().to(self.device)
        self.Critic1_target = Net.Critic().to(self.device)
        self.Critic2_target = Net.Critic().to(self.device)
        self.critierion = nn.MSELoss()
        self.optim_Policy = torch.optim.Adam(self.Policy.parameters(),lr=4e-4)
        self.optim_Critic1 = torch.optim.Adam(self.Critic1.parameters(),lr=4e-4)
        self.optim_Critic2 = torch.optim.Adam(self.Critic2.parameters(),lr=4e-4)
        self.log_alpha = torch.tensor(np.log(0.01),dtype=torch.float,device=self.device)
        self.log_alpha.requires_grad = True
        self.optim_alpha = torch.optim.Adam([self.log_alpha],lr=4e-4)
        self.Policy , self.optim_Policy = amp.initialize(self.Policy,self.optim_Policy,opt_level='O0')
        self.Critic1 , self.optim_Critic1 = amp.initialize(self.Critic1,self.optim_Critic1,opt_level='O0')
        self.Critic2 , self.optim_Critic2 = amp.initialize(self.Critic2,self.optim_Critic2,opt_level='O0')
        # self.log_alpha , self.optim_alpha = amp.initialize(self.log_alpha,self.optim_alpha)
        self.Critic1_target.load_state_dict(self.Critic1.state_dict())
        self.Critic2_target.load_state_dict(self.Critic2.state_dict())
        self.gamma = 0.98
        self.tau = 0.01
        self.target_entropy = -8

    def take_action(self,state):
        state = torch.tensor(np.array([state]),dtype=torch.float).to(self.device)
        return self.Policy(state)[0].detach().cpu().numpy().flatten()
    
    def calc_target(self,reward,next_state,done):
        next_action , log_porb = self.Policy(next_state)
        entropy = -log_porb
        q1 = self.Critic1_target(next_state,next_action)
        q2 = self.Critic2_target(next_state,next_action)
        next_value = torch.min(q1,q2)+self.log_alpha.exp()*entropy
        td_target = reward+self.gamma*next_value*(1-done)
        return td_target
    
    def soft_update(self,net,target_net):
        for param_target , param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data*(1-self.tau)+param.data*self.tau)

    def update(self,states,actions,rewards,next_states,dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device=self.device)
        dones = torch.LongTensor(np.array(dones)).to(device=self.device)
        td_target = self.calc_target(rewards,next_states,dones).detach()
        critic1_loss = torch.mean(self.critierion(self.Critic1(states,actions),td_target))
        critic2_loss = torch.mean(self.critierion(self.Critic2(states,actions),td_target))
        # print(critic1_loss,critic2_loss)
        self.optim_Critic1.zero_grad()
        self.optim_Critic2.zero_grad()
        # critic1_loss.backward()
        # critic2_loss.backward()
        with amp.scale_loss(critic1_loss,self.optim_Critic1) as scaled_loss:
            scaled_loss.backward()
        with amp.scale_loss(critic2_loss,self.optim_Critic2) as scaled_loss:
            scaled_loss.backward()
        self.optim_Critic1.step()
        self.optim_Critic2.step()
        new_actions , log_probs = self.Policy(states)
        entropy = -log_probs
        q1_value = self.Critic1(states,new_actions)
        q2_value = self.Critic2(states,new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp()*entropy-torch.min(q1_value,q2_value))
        # print(f'actor loss:{actor_loss}')
        self.optim_Policy.zero_grad()
        # actor_loss.backward()
        with amp.scale_loss(actor_loss,self.optim_Policy) as scaled_loss:
            scaled_loss.backward()
        self.optim_Policy.step()
        alpha_loss = torch.mean((entropy-self.target_entropy).detach()*self.log_alpha.exp())
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        # with amp.scale_loss(alpha_loss,self.optim_alpha) as scaled_loss:
        #     scaled_loss.backward()
        self.optim_alpha.step()
        self.soft_update(self.Critic1,self.Critic1_target)
        self.soft_update(self.Critic2,self.Critic2_target)

    def save(self):
        torch.save([self.Policy.state_dict(),self.Critic1.state_dict(),self.Critic2.state_dict(),self.log_alpha],'model.pth')
    
    def load(self):
        policy , critic1 , critic2 , self.log_alpha = torch.load('model.pth')
        self.Policy.load_state_dict(policy)
        self.Critic1.load_state_dict(critic1)
        self.Critic2.load_state_dict(critic2)
        self.Critic1_target.load_state_dict(critic1)
        self.Critic2_target.load_state_dict(critic2)
        self.optim_alpha = torch.optim.Adam([self.log_alpha],lr=4e-4)
        self.optim_Critic1 = torch.optim.Adam(self.Critic1.parameters(),lr=4e-4)
        self.optim_Critic2 = torch.optim.Adam(self.Critic2.parameters(),lr=4e-4)
        self.optim_Policy = torch.optim.Adam(self.Policy.parameters(),lr=4e-4)
        self.Policy , self.optim_Policy = amp.initialize(self.Policy,self.optim_Policy,opt_level='O0')
        self.Critic1 , self.optim_Critic1 = amp.initialize(self.Critic1,self.optim_Critic1,opt_level='O0')
        self.Critic2 , self.optim_Critic2 = amp.initialize(self.Critic2,self.optim_Critic2,opt_level='O0')
    # def judge_out_of_route(self, obs):
    #     s = obs[:84, 6:90, :]
    #     out_sum = (s[75, 35:48, 1][:2] > 200).sum() + (s[75, 35:48, 1][-2:] > 200).sum()
    #     return out_sum == 4