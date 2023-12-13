import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

data = pd.read_csv("./data.csv")
print(data.shape)#(943, 22)
#data.columns
movie = data.iloc[:, 1:]
action_space= movie.iloc[:, -1].unique().tolist()

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1 # or use 0.1

#The number of iterations (epochs) for updating the neural network
K_epoch = 10

#policy will be updated based on the data collected over the T_horizon time steps
T_horizon = 20

epsilon = 0.2


#create the environment
class CustomEnvironment:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.step = 0
        self.num_states = len(self.dataset) #943
        self.state_size = 19 #19 features
        self.action_space = action_space
        self.model = model  # add ppo model as an argument

    def reset(self):
        self.step = 0
        return self.get_state()

    def get_state(self):  # retrieve the current state
        current_row = self.dataset.iloc[self.step]
        current_label = current_row[['19']].values.item()  # change the value of Y from array to integer

        #s = current_row[['userId', '0', '1', '2', '3', '4', '5', '6', '7', '8',
       #'9', '10', '11', '12', '13', '14', '15', '16', '17', '18' ]].values # state is an array
        # kaichuang zhang modification
        s = current_row[['0', '1', '2', '3', '4', '5', '6', '7', '8',
        '9', '10', '11', '12', '13', '14', '15', '16', '17', '18' ]].values # state is an array
        return s



    def sample_action(self, s, epsilon):#To obtain the log probability of acutal action a
        state = torch.from_numpy(s).float().unsqueeze(0)
        probs = self.model.pi(state)

        coin = random.random()
        if coin < epsilon:
           #a = random.choice(action_space)
           a = random.randint(0, 430)
           # kaichuang zhang modification
           prob_a = 1/431 * torch.ones_like(probs[0, 1]) # Create a tensor value of 1/431
           #log_prob = torch.log(prob)
           #log_prob = torch.tensor([0.000001]) # To avoid empty tensors, specifically tensor([0.])##
        else :
           a = torch.argmax(probs).item()
           # Use item() to get the scalar value from the tensor. Random case: a = torch.multinomial(probs, 1).item()
           prob_a = probs[0, a] # or use prob = probs[0][a] to get the probability of the selected action
           #prob_a = probs
           #log_prob = torch.log(prob) # Obtain the log probability of the sampled action
        return a, prob_a


    def get_next_state(self):
        if self.step < self.num_states - 1:
           next_row = self.dataset.iloc[self.step + 1]
           #s_prime = next_row[['userId', '0', '1', '2', '3', '4', '5', '6', '7', '8',
       #'9', '10', '11', '12', '13', '14', '15', '16', '17', '18' ]].values
           s_prime = next_row[['0', '1', '2', '3', '4', '5', '6', '7', '8',
        '9', '10', '11', '12', '13', '14', '15', '16', '17', '18' ]].values
        else: # If self.step is already at the last row, reset

           s_prime = np.zeros(19) # use all zeros as the last row

        return s_prime


    def get_reward(self, epsilon):

        current_row = self.dataset.iloc[self.step]
        current_label = current_row[['19']].values.item()#change the value of Y from array to integer

        if self.step < self.num_states - 1:
           done = False
           action, _ = self.sample_action(self.get_state(), epsilon)  # Pass state and epsilon

        #Calculate the reward
           if action == current_label:
              r = 1  # Correct classification
           else:
              r = 0  # Incorrect classification


        else:
          # If self.step is already at the last row, reset
            self.reset()
            done = True
            r = 0

        return r, done
    
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []#A list to store transition data
        # nn.Embedding()
        self.fc1 = nn.Linear(19, 256)
        self.fc_pi = nn.Linear(256, 431)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s.tolist())
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime.tolist())
            prob_a_lst.append([prob_a.tolist()])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        #print (f's_lst : {s_lst}')
        #print (f'a_lst : {a_lst}')
        #print (f'r_lst : {r_lst}')
        #print (f's_prime_lst: {s_prime_lst}')
        #print (f'done_lst :{done_lst}')
        #print (f'prob_a_lst: {prob_a_lst}')
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):#32 in this case
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]: #Iterate over the inverse order from the most recent time
                advantage = gamma * lmbda * advantage + delta_t[0]#Compute the GAE
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)#The probability of acutal action
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            print (f'surr1 {surr1}, {surr2}')
            print (f'loss {self.v(s)}, {td_target}')
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())
            print (f'loss {loss}')

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
def main():
    model = PPO()
    env = CustomEnvironment(movie, model)  # add model

    score = 0.0
    print_interval = 10
    scores = []  # To store average scores for each episode
    episodes = []

    for n_epi in range(1000):
        s= env.reset()
        done = False
        while not done:
            for t in range(T_horizon):

                s = env.get_state()
                s_prime = env.get_next_state()
                r, done = env.get_reward(epsilon)
                a, prob_a = env.sample_action(s, epsilon)
                #print(s, a, r, s_prime, log_prob)

                #prob = model.pi(torch.from_numpy(s).float())
                #m = Categorical(prob)
                #a = m.sample().item()


                model.put_data((s, a, r , s_prime, prob_a, done))
                s = s_prime
                env.step += 1

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
            scores.append(avg_score)
            episodes.append(n_epi)
            score = 0.0

    #env.close()

    # Plot the learning curve
    plt.plot(episodes, scores)
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.title('Learning Curve')
    plt.savefig("./score.png")
    plt.show()

    # Save the model
    torch.save(model, 'model_PPO.pth')

if __name__ == "__main__":
    main()