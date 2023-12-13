from metrics.Metrics import Metrics
import numpy as np

class NDCG(Metrics):
    # reference : https://blog.csdn.net/baishuiniyaonulia/article/details/119761820
    #             https://www.cnblogs.com/gczr/p/15007615.html
    def __init__(self, prediction, groundtruth) -> None:
        super().__init__(prediction, groundtruth)


    def DCG(self, user_prediction, user_groundtruth):
        """compute the ncg

        Returns:
            flaot: dcg
        """
        dcg = 0.
        for i in range(len(user_prediction)):
            r_i = 0
            if user_prediction[i] in user_groundtruth:
                r_i = 1
            dcg += (2 ** r_i - 1) / np.log2((i + 1) + 1) # (i+1)是因为下标从0开始
        return dcg

    def IDCG(self, user_prediction, user_groundtruth):
        """compute the idcg

        Args:
            user_prediction (list): the recommend items
            user_groundtruth (list): the ground true items

        Returns:
            flaot: idcg
        """
        # ------ 将在测试中的a排到前面去，然后再计算DCG ------ #
        A_temp_1 = [] # 临时A，用于存储r_i为1的a
        A_temp_0 = []  # 临时A，用于存储r_i为0的a
        for a in user_prediction:
            if a in user_groundtruth:
                # 若a在测试集中则追加到A_temp_1中
                A_temp_1.append(a)
            else:
                # 若a不在测试集中则追加到A_temp_0中
                A_temp_0.append(a)
        A_temp_1.extend(A_temp_0)
        idcg = self.DCG(A_temp_1, user_groundtruth)
        return idcg
    
    def NDCG(self, user_prediction, user_groundtruth):
        dcg = self.DCG(user_prediction, user_groundtruth) # 计算DCG
        idcg = self.IDCG(user_prediction, user_groundtruth) # 计算IDCG
        if dcg == 0 or idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg
        return ndcg
    
    def compute(self):
        sum_ndcg = 0.
        for user_prediction, user_groundtruth in zip(self.prediction, self.groundtruth):
            sum_ndcg += self.NDCG(user_prediction, user_groundtruth)
        return sum_ndcg / self.prediction_len

if __name__ == "__main__":
    # ------ 计算推荐列表A的NDCG ------ #
    # A：推荐列表，一维list，存储了推荐算法推荐出的推荐项的id
    # test_set：测试集，一维list，存储了测试集推荐项的id
    recommended_items = [[1, 6, 7], [2, 3, 4, 5], [1, 2, 3]]
    true_preferences = [[1, 2, 3, 4, 5], [2, 3, 6, 7], [1, 4, 8]]
    ndcg_A = NDCG(recommended_items, true_preferences)
    print (f"NDCG : {ndcg_A.compute()}")