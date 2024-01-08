from collections import defaultdict

class LambdaDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory(key)
