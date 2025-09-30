class BisectionMethodOptimizer:

    def __init__(self, min: float, max: float, n_iters: int, direction: str = "maximize"):
        self.low = min
        self.high = max
        self.n_iters = n_iters
        self.delta = 0.01
        self.direction = direction

    def run(self, obj_func):

        for i in range(self.n_iters):

            mid = (self.low + self.high) / 2
            left = mid - self.delta
            right = mid + self.delta

            _, _, f_mid = obj_func(anomaly_threshold=mid)
            _, _, f_left = obj_func(anomaly_threshold=left)
            _, _, f_right = obj_func(anomaly_threshold=right)

            if self.direction == "maximize":

                if f_left > f_mid:
                    self.high = mid
                elif f_right > f_mid:
                    self.low = mid
                else:
                    self.low = left
                    self.high = right

            else:

                if f_left < f_mid:
                    self.high = mid
                elif f_right < f_mid:
                    self.low = mid
                else:
                    self.low = left
                    self.high = right

        best_x = (self.high + self.low) / 2
        best_y = obj_func(best_x)

        return best_x, best_y



