from Metrics import Metrics

class MAP(Metrics):
    # reference : https://blog.csdn.net/xiedelong/article/details/112500657
    def __init__(self, prediction, groundtruth) -> None:
        super().__init__(prediction, groundtruth)

    def user_ap(self, prediction, groundtruth):
        """compute the ap

        Args:
            prediction (list): predict items
            groundtruth (list): true items

        Returns:
            float: ap of user
        """
        hits = 0
        sum_precs = 0
        for n in range(len(prediction)):
            if prediction[n] in groundtruth:
                hits += 1
                sum_precs += hits / (n + 1.0)
        if hits > 0:
            return sum_precs / len(groundtruth)
        else:
            return 0

    def average_ap(self):
        """return the mean ap 
        """
        sum_ap = 0.
        for user_prediction, truth in zip(self.prediction, self.groundtruth):
            sum_ap += self.user_ap(user_prediction, truth)
        return sum_ap / len(self.prediction)

    def compute(self):
        return self.average_ap()
    

if __name__ == '__main__':
    recommended_items = [[1, 6, 7], [2, 3, 4, 5], [1, 2, 3]]
    true_preferences = [[1, 2, 3, 4, 5], [2, 3, 6, 7], [1, 4, 8]]
    map_calculator = MAP(recommended_items, true_preferences)
    print (f"mean ap : {map_calculator.compute()}")