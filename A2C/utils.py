import pandas as pd
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from replay_memory import ReplayMemory
import time


def read_file(data_path):
  ''' Load data from train.csv or test.csv. '''

  data = pd.read_csv(data_path, sep=';')
  for col in ['state', 'n_state', 'action_reward']:
    data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
  # data[state] = [[movieid, reward], [movieid, reword], []....]
  # data[n_state] = [[movieid, reword], [movieid, reward], ]
  # data[action_reward] =[[movieid, reword], [movieid, reward], ]
  for col in ['state', 'n_state']:
    data[col] = [np.array([e[0] for e in l]) for l in data[col]]
  # data[state] = [movieid, movieid, movieid...] delete the rewawrd
  # data[n_state] = [movied, movieid ....] 

  data['action'] = [[e[0] for e in l] for l in data['action_reward']]
  data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
  # date[action] = [movieid, movieid ...]
  # data[reward] = [rating, rating, rating...]
  data.drop(columns=['action_reward'], inplace=True)
  # 删掉 action_rewarrd
  # data[state] = [movieid, movieid, movieid...]
  # data[n_state] = [movieid, movieid, movieid...]
  # data[action] = [movieid, movieid, movieid...]
  # data[reward] = [rating, rating, rating...]
  print (data)
  return data

def experience_replay(replay_memory, batch_size, actor, critic, embeddings, ra_length, state_space_size, action_space_size, discount_factor):
  '''
  Experience replay.
  Args:
    replay_memory: replay memory D in article.
    batch_size: sample size.
    actor: Actor network.
    critic: Critic network.
    embeddings: Embeddings object.
    state_space_size: dimension of states.
    action_space_size: dimensions of actions.
  Returns:
    Best Q-value, loss of Critic network for printing/recording purpose.
  '''

  # '22: Sample minibatch of N transitions (s, a, r, s′) from D'
  samples = replay_memory.sample_batch(batch_size)
  states = np.array([s[0] for s in samples])
  actions = np.array([s[1] for s in samples])
  rewards = np.array([s[2] for s in samples])
  n_states = np.array([s[3] for s in samples]).reshape(-1, state_space_size)

  # '23: Generate a′ by target Actor network according to Algorithm 2'
  n_actions = actor.get_recommendation_list(ra_length, states, embeddings, target=True).reshape(-1, action_space_size)

  # Calculate predicted Q′(s′, a′|θ^µ′) value
  target_Q_value = critic.predict_target(n_states, n_actions, [ra_length] * batch_size)

  # '24: Set y = r + γQ′(s′, a′|θ^µ′)'
  expected_rewards = rewards + discount_factor * target_Q_value
  
  # '25: Update Critic by minimizing (y − Q(s, a|θ^µ))²'
  critic_Q_value, critic_loss, _ = critic.train(states, actions, [ra_length] * batch_size, expected_rewards)
  
  # '26: Update the Actor using the sampled policy gradient'
  action_gradients = critic.get_action_gradients(states, n_actions, [ra_length] * batch_size)
  actor.train(states, [ra_length] * batch_size, action_gradients)

  # '27: Update the Critic target networks'
  critic.update_target_network()

  # '28: Update the Actor target network'
  actor.update_target_network()

  return np.amax(critic_Q_value), critic_loss

def train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary, nb_rounds=50):
  ''' Algorithm 3 in article. '''

  # Set up summary operators
  def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar('reward', episode_reward)
    episode_max_Q = tf.Variable(0.)
    tf.summary.scalar('max_Q_value', episode_max_Q)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar('critic_loss', critic_loss)

    summary_vars = [episode_reward, episode_max_Q, critic_loss]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

  summary_ops, summary_vars = build_summaries()
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(filename_summary, sess.graph)

  # '2: Initialize target network f′ and Q′'
  # init all the network;
  actor.init_target_network()
  critic.init_target_network()

  # '3: Initialize the capacity of replay memory D'
  # init replay memory
  replay_memory = ReplayMemory(buffer_size) # Memory D in article
  replay = False


  start_time = time.time()
  for i_session in range(nb_episodes): # '4: for session = 1, M do'
    session_reward = 0
    session_Q_value = 0
    session_critic_loss = 0

    # '5: Reset the item space I' is useless because unchanged.

    states = environment.reset() # '6: Initialize state s_0 from previous sessions'
    
    if (i_session + 1) % 10 == 0: # Update average parameters every 10 episodes
      environment.groups = environment.get_groups()
      
    exploration_noise = OrnsteinUhlenbeckNoise(history_length * embeddings.size())

    for t in range(nb_rounds): # '7: for t = 1, T do'
      # '8: Stage 1: Transition Generating Stage'

      # '9: Select an action a_t = {a_t^1, ..., a_t^K} according to Algorithm 2'
      actions = actor.get_recommendation_list(
          ra_length,
          states.reshape(1, -1), # TODO + exploration_noise.get().reshape(1, -1),
          embeddings).reshape(ra_length, embeddings.size())

      # '10: Execute action a_t and observe the reward list {r_t^1, ..., r_t^K} for each item in a_t'
      rewards, next_states = environment.step(actions)

      # '19: Store transition (s_t, a_t, r_t, s_t+1) in D'
      replay_memory.add(states.reshape(history_length * embeddings.size()),
                        actions.reshape(ra_length * embeddings.size()),
                        [rewards],
                        next_states.reshape(history_length * embeddings.size()))

      states = next_states # '20: Set s_t = s_t+1'

      session_reward += rewards
      
      # '21: Stage 2: Parameter Updating Stage'
      if replay_memory.size() >= batch_size: # Experience replay
        replay = True
        replay_Q_value, critic_loss = experience_replay(replay_memory, batch_size,
          actor, critic, embeddings, ra_length, history_length * embeddings.size(),
          ra_length * embeddings.size(), discount_factor)
        session_Q_value += replay_Q_value
        session_critic_loss += critic_loss

      summary_str = sess.run(summary_ops,
                             feed_dict={summary_vars[0]: session_reward,
                                        summary_vars[1]: session_Q_value,
                                        summary_vars[2]: session_critic_loss})
      
      writer.add_summary(summary_str, i_session)

      '''
      print(state_to_items(embeddings.embed(data['state'][0]), actor, ra_length, embeddings),
            state_to_items(embeddings.embed(data['state'][0]), actor, ra_length, embeddings, True))
      '''

    str_loss = str('Loss=%0.4f' % session_critic_loss)
    print(('Episode %d/%d Reward=%d Time=%ds ' + (str_loss if replay else 'No replay')) % (i_session + 1, nb_episodes, session_reward, time.time() - start_time))
    start_time = time.time()

  writer.close()
  tf.train.Saver().save(sess, 'models.h5', write_meta_graph=False)

class OrnsteinUhlenbeckNoise:
  ''' Noise for Actor predictions. '''
  def __init__(self, action_space_size, mu=0, theta=0.5, sigma=0.2):
    self.action_space_size = action_space_size
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.state = np.ones(self.action_space_size) * self.mu

  def get(self):
    self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.rand(self.action_space_size)
    return self.state