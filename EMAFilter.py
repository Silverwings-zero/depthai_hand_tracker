class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_value = None

    def __call__(self, value):
        if self.prev_value is None:
            self.prev_value = value
        else:
            self.prev_value = self.alpha * value + (1 - self.alpha) * self.prev_value
        return self.prev_value