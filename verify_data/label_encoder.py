
class LabelEncoder:
    def __init__(self):
        self.label_idx = {}
        self.counter = 0

    def fit(self, y):
        """Fit the date to the class

        Iterate through all lines and save the labels mapping to a counter integer."""

        y = y.dropna()
        for idx, line in y.iterrows():
            if list(line)[0] not in self.label_idx:
                self.label_idx[list(line)[0]] = self.counter
                self.counter += 1

    def transform(self, y):
        """Transform the data fitted in the class

        Iterate through all lines keeping the integer of each label.
        :return List: List with all corresponding integers"""

        y = y.dropna()
        return [self.label_idx[list(line)[0]] for idx, line in y.iterrows()]

    def fit_transform(self, y):
        """Execute both the methods at once to simplify the use"""

        self.fit(y)
        return self.transform(y)
