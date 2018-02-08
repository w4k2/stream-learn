class Bare(object):
    def __init__(self):
        self.learner = None

    def prepare(self):
        pass

    def next_chunk(self):
        pass

    def get_measures(self):
        return 0

    def should_include(self, _X, _x, _label):
        return True

    def should_break_chunk(self, _X):
        return False

    def __str__(self):
        return 'bare'
