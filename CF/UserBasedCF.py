from dataset.movielensdataset import MovieLensDataSet
from metrics.MAP import MAP
from metrics.NDCG import NDCG
from Algo import Algorithm

class UserBasedCF(Algorithm):
    # User-Based Collaborative Filtering
    def __init__(self, dataset):
        """
        """
        self.dataset = dataset

    def compute_distance(self, userid1, userid2):
        """
        compute the distance from userid1 to userid2
        """
        # self.train_dataset_np = self.train_dataset.values
        self.user_len = self.train_dataset_np.shape[0]
        # self.test_dataset_np = self.test_dataset.values
        self.distance_matrix = np.zeros((user_len, user_len))

        for i in range(user_len):
            for j in range(user_len):
                print (x, y)
                x = self.train_dataset.iloc(i)
                y = self.train_dataset.iloc(j)
                print (x, y)


    def recommend(self, userid, n = 5, k = 4):
        """
        recommend to user the movies top n
        return :
        the list of recommendations for userid
        """
        similar_userids = self.find_similar_userids(userid)
        if len(similar_userids) == 0:
            return []
        #print ("simalar userid", similar_userids)
        similar_userids_movies = self.find_simiar_userid_movies(userid, similar_userids)
        #print ("similar user movies", similar_userids_movies)
        return similar_userids_movies


    def find_similar_userids(self, userid, n = 5):
        """
        find the silimar user top n
        """
        if userid not in self.dataset.train_dataset:
            return []
        similar_list = []
        for userid_, movies in self.dataset.train_dataset.items():
            if userid_ == userid:
                continue
            similar_list.append((userid_, len(self.dataset.train_dataset[userid].intersection(self.dataset.train_dataset[userid_]))))
        similar_sorted_list = sorted(similar_list, key=lambda x: x[1], reverse=True)
        return [x[0] for x in similar_sorted_list[:n]]

    def find_simiar_userid_movies(self, userid, similar_userids, k = 7):
        """
        find the movies your similar usrids
        """
        movies = {}
        for userid_ in similar_userids:
            for movie in self.dataset.train_dataset[userid_]:
                if movie in self.dataset.train_dataset[userid]:
                    continue
                if movie not in movies:
                    movies[movie] = 0
                movies[movie] += 1
        movies_sorted = sorted(movies.items(), key=lambda x: x[1], reverse=True)
        #print (movies_sorted)
        return [x[0] for x in movies_sorted[:k]]

    def test_model(self):
        """
        return :
        https://www.cnblogs.com/fuxuemingzhu/p/15436050.html
        Precision
        Recall
        Coverage
        Popularity
        """
        # recall_movies_cnt = 0
        # true_movies_cnt = 0
        # hit_movies_cnt = 0
        # print_log_cnt = 100
        # process_cnt = 1
        # recommend_movies_total = {}
        prediction = []
        groundtruth = []
        for userid, movies in self.dataset.test_dataset.items():
            recommend_movies = self.recommend(userid)
            prediction.append(recommend_movies)
            groundtruth.append(movies)
            #if len(recommend_movies) == 0:
            #    continue
            
            """
                for recommend_movie in recommend_movies:
                    if recommend_movie not in recommend_movies_total:
                        recommend_movies_total[recommend_movie] = True
                recall_movies_cnt += len(recommend_movies)
                true_movies_cnt += len(movies)
                userid_hit_movies = set(recommend_movies).intersection(movies)
                hit_movies_cnt += len(userid_hit_movies)
                if process_cnt % print_log_cnt == 0:
                    print (f"process user {process_cnt}, and result:\n precision: {hit_movies_cnt / recall_movies_cnt:.4f}, recall: {hit_movies_cnt / true_movies_cnt:.4f}, converage : {len(recommend_movies_total) / self.dataset.get_movies_cnt():.4f}\n")
                process_cnt += 1
            """
        map_calculator = MAP(prediction, groundtruth)
        ndcg_calculator = NDCG(prediction, groundtruth)
        
        print (f"MAP : {map_calculator.compute()}, NDCG : {ndcg_calculator.compute()}.")
        #print (f"process done result:\n precision: {hit_movies_cnt / recall_movies_cnt:.4f}, recall: {hit_movies_cnt / true_movies_cnt:.4f}, converage : {len(recommend_movies_total) / self.dataset.get_movies_cnt():.4f}\n")
if __name__ == '__main__':
    DATA_DIR = "./ml-100k/"
    ml_dataset = MovieLensDataSet(DATA_DIR)
    userbased_cf = UserBasedCF(ml_dataset)
    userbased_cf.test_model()
