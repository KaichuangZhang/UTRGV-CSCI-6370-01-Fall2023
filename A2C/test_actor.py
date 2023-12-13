import numpy as np

from MAP import MAP
from NDCG import NDCG

def state_to_items(state, actor, ra_length, embeddings, dict_embeddings, target=False):
  return [dict_embeddings[str(action)]
          for action in actor.get_recommendation_list(ra_length, np.array(state).reshape(1, -1), embeddings, target).reshape(ra_length, embeddings.size())]

def test_actor(actor, test_df, embeddings, dict_embeddings, ra_length, history_length, target=False, nb_rounds=1):
    prediction = []
    groundtruth = []
    for _ in range(nb_rounds):
        for i in range(len(test_df)):
            n = len(test_df[i])
            #history_samples = list(test_df[i].head(int(0.8 * n)).sample(history_length)['itemId'])
            history_sample = list(test_df[i].sample(history_length)['itemId']) # sample histroy_length.
            ground_true_samples = list(test_df[i]['itemId'])
            recommendation = state_to_items(embeddings.embed(history_sample), actor, ra_length, embeddings, dict_embeddings, target)
            prediction.append(recommendation)
            groundtruth.append(ground_true_samples)

    map_c =MAP(prediction, groundtruth)
    ndgc_c = NDCG(prediction, groundtruth)
    print (f'MAP: {map_c.compute()}')
    print (f'NDGC: {ndgc_c.compute()}')

    
