import pickle


class PickleLoader:
    def __init__(self, filename):
        self.filename = 'pickles/' + filename
        try:
            with open(self.filename, "rb") as f:
                self.data = pickle.load(f)
        except:
            self.data = None

    def return_data(self):
        return self.data

    def dump_data(self, data):
        with open(self.filename, "wb+") as f:
            pickle.dump(data, f)