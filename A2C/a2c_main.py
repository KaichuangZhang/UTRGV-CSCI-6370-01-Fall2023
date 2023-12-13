from data_generator import DataGenerator
from embeddings_generator import EmbeddingsGenerator, Embeddings, read_embeddings
from utils import read_file, train
import pandas as pd
from actor import Actor
from critic import Critic
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from simulator import Environment
#from test import test_actor
from test_actor import test_actor


# Hyperparameters
history_length = 15 # N in article
ra_length = 4 # K in article
discount_factor = 0.99 # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001 # τ in Algorithm 3
batch_size = 128
nb_episodes = 50
nb_rounds = 50
filename_summary = 'summary.txt'
alpha = 0.5 # α (alpha) in Equation (1)
gamma = 0.9 # Γ (Gamma) in Equation (4)
buffer_size = 1000000 # Size of replay memory D in article
fixed_length = True # Fixed memory length

if __name__ == '__main__':
    # read the data
    dg = DataGenerator('../dataset/ml-100k/u.data', '../dataset/ml-100k/u.item')
    dg.gen_train_test(0.8, seed=42)

    dg.write_csv('train.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
    dg.write_csv('test.csv', dg.test, nb_states=[history_length], nb_actions=[ra_length])

    data = read_file('train.csv')


    if True:
        eg = EmbeddingsGenerator(dg.user_train, pd.read_csv('../dataset/ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']))
        eg.train(nb_epochs=10)
        train_loss, train_accuracy = eg.test(dg.user_train)
        print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
        test_loss, test_accuracy = eg.test(dg.user_test)
        print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
        eg.save_embeddings('embeddings.csv')

    # reduce the size of state, and  action
    embeddings = Embeddings(read_embeddings('embeddings.csv'))

    state_space_size = embeddings.size() * history_length
    action_space_size = embeddings.size() * ra_length

    print (f'state space size : {state_space_size}, action_space_size {action_space_size}')
    environment = Environment(data, embeddings, alpha, gamma, fixed_length)

    tf.reset_default_graph() # For multiple consecutive executions

    sess = tf.Session()
    # '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
    actor = Actor(sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embeddings.size(), tau, actor_lr)
    critic = Critic(sess, state_space_size, action_space_size, history_length, embeddings.size(), tau, critic_lr)

    train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary)

    if True: # for test
        dict_embeddings = {}
        for i, item in enumerate(embeddings.get_embedding_vector()):
            str_item = str(item)
            assert(str_item not in dict_embeddings)
            dict_embeddings[str_item] = i

        test_actor(actor, dg.train, embeddings, dict_embeddings, ra_length, history_length, target=True, nb_rounds=10)
