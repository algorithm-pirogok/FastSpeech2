class BaseMetric:
    def __init__(self, name=None, train=True, mode: str = 'argmax', *args, **kwargs):
        self.train = train
        self.mode = mode
        self.name = name if name is not None else type(self).__name__

    def __call__(self, **batch):
        raise NotImplementedError()
