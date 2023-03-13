import os


class Example:
    def __init__(self):
        self.experiment = 'test'
        print(os.getcwd())

    def make_save_name(self):
        path = os.path.join('Library', self.experiment)
        print(path)
