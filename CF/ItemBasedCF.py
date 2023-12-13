from metrics.MAP import MAP
from metrics.NDCG import NDCG
from Algo import Algorithm

class ItemBasedCF(Algorithm):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.movie_similarity_cache = dict()

    def get_movie_similarity(self, movieid):
        """get the similarity of movie with movieid

        Args:
            movieid (string): the movie you want to get the silimarity
            k (int, optional): _description_. Defaults to 7.
        """
        if movieid in self.movie_similarity_cache:
            return self.movie_similarity_cache[movieid]
        else:
            self.movie_similarity_cache[movieid] = []
        movieid_like_userids = set(self.dataset.movies[movieid])
        for movie_id, like_userids in self.dataset.movies.items():
            simliarity = len(movieid_like_userids.intersection(set(like_userids)))
            self.movie_similarity_cache[movieid].append([movie_id, simliarity])
            # len(self.dataset.train_dataset[userid].intersection(self.dataset.train_dataset[userid_]))))
        self.movie_similarity_cache[movieid].sort(key=lambda x: x[1], reverse=True)
        self.movie_similarity_cache[movieid] = self.movie_similarity_cache[movieid]
        return self.movie_similarity_cache[movieid] 
        
        
    def recommend(self, userid, k = 4):
        """recommend movies for userid

        Args:
            userid (string): userid
        """
        if userid not in self.dataset.train_dataset:
            return []
        recommend_result = dict()
        for movieid in self.dataset.train_dataset[userid]:
            # get the similar dataset
            movies = self.get_movie_similarity(movieid)
            for movie, similarity in movies:
                if movie not in recommend_result:
                    recommend_result[movie] = 0
                recommend_result[movie] += similarity
        recommend_result_list = list(recommend_result.items())
        recommend_result_list.sort(key=lambda x: x[1], reverse=True)
        #print (recommend_result_list[:k])
        return [x[0] for x in recommend_result_list[:k]]


    def test_model(self):
        """test the model
        """
        prediction = []
        groundtruth = []
        for userid, movies in self.dataset.test_dataset.items():
            recommend_movies = self.recommend(userid)
            #print (f"userid: {userid}, recommend : {recommend_movies}")
            prediction.append(recommend_movies)
            groundtruth.append(movies)
        
        map_calculator = MAP(prediction, groundtruth)
        ndcg_calculator = NDCG(prediction, groundtruth)
        
        print (f"MAP : {map_calculator.compute()}, NDCG : {ndcg_calculator.compute()}.")
        