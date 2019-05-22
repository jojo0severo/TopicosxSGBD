

class LabelEncoder:
    def __init__(self):
        self.label_idx = {}
        self.counter = 0

    def fit(self, y):
        y = y.dropna()
        for idx, line in y.iterrows():
            if list(line)[0] not in self.label_idx:
                self.label_idx[list(line)[0]] = self.counter
                self.counter += 1

    def transform(self, y):
        y = y.dropna()
        return [self.label_idx[list(line)[0]] for idx, line in y.iterrows()]

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
