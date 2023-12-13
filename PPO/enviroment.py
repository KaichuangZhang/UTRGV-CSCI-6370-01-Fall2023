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