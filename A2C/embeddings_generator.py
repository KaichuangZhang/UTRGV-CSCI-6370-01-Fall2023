from keras import Sequential
import keras.backend as K
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np


class EmbeddingsGenerator:
  def  __init__(self, train_users, data):
    self.train_users = train_users # train data

    #preprocess
    self.data = data.sort_values(by=['timestamp']) # 按时间排序的打分数据
    #make them start at 0
    self.data['userId'] = self.data['userId'] - 1
    self.data['itemId'] = self.data['itemId'] - 1
    self.user_count = self.data['userId'].max() + 1 # 用户数据
    self.movie_count = self.data['itemId'].max() + 1 # movie数据
    self.user_movies = {} #list of rated movies by each user
    for userId in range(self.user_count): # user - ratings
      self.user_movies[userId] = self.data[self.data.userId == userId]['itemId'].tolist()
    self.m = self.model()

  def model(self, hidden_layer_size=100): # 这是模型?
    m = Sequential()
    m.add(Dense(hidden_layer_size, input_shape=(1, self.movie_count)))
    m.add(Dropout(0.2))
    m.add(Dense(self.movie_count, activation='softmax'))
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m
  
  def generate_input(self, user_id):
    '''
    Returns a context and a target for the user_id
    context: user's history with one random movie removed
    target: id of random removed movie
    '''
    user_movies_count = len(self.user_movies[user_id])
    #picking random movie
    random_index = np.random.randint(0, user_movies_count-1) # -1 avoids taking the last movie
    #setting target
    target = np.zeros((1, self.movie_count))
    target[0][self.user_movies[user_id][random_index]] = 1
    #setting context
    context = np.zeros((1, self.movie_count))
    context[0][self.user_movies[user_id][:random_index] + self.user_movies[user_id][random_index+1:]] = 1
    return context, target

  def train(self, nb_epochs = 300, batch_size = 10000):
    '''
    Trains the model from train_users's history
    '''
    for i in range(nb_epochs):
      print('%d/%d' % (i+1, nb_epochs))
      # 选择一个batch_size来训练模型
      batch = [self.generate_input(user_id=np.random.choice(self.train_users) - 1) for _ in range(batch_size)]
      X_train = np.array([b[0] for b in batch])
      y_train = np.array([b[1] for b in batch])
      self.m.fit(X_train, y_train, epochs=1, validation_split=0.5)

  def test(self, test_users, batch_size = 100000):
    '''
    Returns [loss, accuracy] on the test set
    '''
    batch_test = [self.generate_input(user_id=np.random.choice(test_users) - 1) for _ in range(batch_size)]
    X_test = np.array([b[0] for b in batch_test])
    y_test = np.array([b[1] for b in batch_test])
    return self.m.evaluate(X_test, y_test)

  def save_embeddings(self, file_name):
    '''
    Generates a csv file containg the vector embedding for each movie.
    '''
    inp = self.m.input                                           # input placeholder
    outputs = [layer.output for layer in self.m.layers]          # all layer outputs
    #functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function
    functor = K.function([inp], outputs )   # evaluation function

    #append embeddings to vectors
    vectors = []
    for movie_id in range(self.movie_count):
      movie = np.zeros((1, 1, self.movie_count))
      movie[0][0][movie_id] = 1
      #print (f'movie type {type(movie)}')
      layer_outs = functor([movie])
      vector = [str(v) for v in layer_outs[0][0][0]]
      vector = '|'.join(vector)
      vectors.append([movie_id, vector])

    #saves as a csv file
    embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'item_id': 'int32'})
    embeddings.to_csv(file_name, sep=';', index=False)
    #files.download(file_name) 


class Embeddings:
  def __init__(self, item_embeddings):
    self.item_embeddings = item_embeddings
  
  def size(self):
    return self.item_embeddings.shape[1]
  
  def get_embedding_vector(self):
    return self.item_embeddings
  
  def get_embedding(self, item_index):
    return self.item_embeddings[item_index]

  def embed(self, item_list):
    return np.array([self.get_embedding(item) for item in item_list])


def read_embeddings(embeddings_path):
  ''' Load embeddings (a vector for each item). '''
  
  embeddings = pd.read_csv(embeddings_path, sep=';')

  return np.array([[np.float64(k) for k in e.split('|')]
                   for e in embeddings['vectors']])
     