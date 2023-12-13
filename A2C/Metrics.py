
class Metrics(object):
    def __init__(self, prediction, groundtruth) -> None:
        self.prediction_len = len(prediction)
        self.groundtruth_len = len(groundtruth)
        assert(self.prediction_len == self.groundtruth_len)
        self.prediction = prediction
        self.groundtruth = groundtruth

    def compute(self):
        pass
