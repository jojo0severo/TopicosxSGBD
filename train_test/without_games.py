import pandas as pd
import random as rn
from verify_data.label_encoder import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


RANDOM_STATE = 30
LABEL_ENCODER = LabelEncoder()

# Load dataset from CSV
df = pd.read_csv('../clean_dataset.csv')
df = df.drop(['games'], axis=1)


# Set seed
rn.seed = RANDOM_STATE


# Select some values for validation
validation = df.sample(frac=.15, random_state=RANDOM_STATE)
df = df.drop(validation.index)
x_validation = validation.iloc[:, 1:]
y_validation = validation.iloc[:, 0]

# Get the train and test data
train_df = df.sample(frac=.8, random_state=RANDOM_STATE)
test_df = df.sample(frac=.2, random_state=RANDOM_STATE)

x_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]

x_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]


# Instantiate models
logistic_regression = LogisticRegression(solver='lbfgs', multi_class='multinomial')

tree_classifier = DecisionTreeClassifier()
extra_tree_classifier = ExtraTreeClassifier()
tree_estimators = [('tc', DecisionTreeClassifier()), ('etc', ExtraTreeClassifier())]
voting_classifier_tree = VotingClassifier(estimators=tree_estimators)

adaboost_classifier = AdaBoostClassifier()
extra_trees_classifier = ExtraTreesClassifier(n_estimators=70)
bagging_classifier = BaggingClassifier()
random_forest_classifier = RandomForestClassifier()
gradient_boost_classifier = GradientBoostingClassifier()
boost_estimators = [
    ('adac', AdaBoostClassifier()), ('etsc', ExtraTreesClassifier(n_estimators=70)),
    ('bc', BaggingClassifier()), ('rfc', RandomForestClassifier()), ('gbc', GradientBoostingClassifier())]
voting_classifier_boost = VotingClassifier(estimators=boost_estimators, voting='soft')

mlp_classifier = MLPClassifier(max_iter=700, random_state=RANDOM_STATE)
gaussian_nb = GaussianNB()
nb_estimators = [('mc', MLPClassifier()), ('gnb', GaussianNB())]
voting_classifier_nb = VotingClassifier(estimators=nb_estimators)

mix_estimators = [
    ('le', LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')),
    ('te', rn.choice(tree_estimators)[1]), *rn.sample(boost_estimators, 4), *rn.sample(nb_estimators, 2)]
voting_classifier_mix = VotingClassifier(estimators=mix_estimators)

all_estimators = [('lgr',  LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')),
                  *tree_estimators, *boost_estimators, *nb_estimators]
voting_classifier_all = VotingClassifier(estimators=all_estimators)


# Train all models, (it will take some time)
logistic_regression.fit(x_train, y_train)

tree_classifier.fit(x_train, y_train)
extra_tree_classifier.fit(x_train, y_train)

adaboost_classifier.fit(x_train, y_train)
extra_trees_classifier.fit(x_train, y_train)
bagging_classifier.fit(x_train, y_train)
gradient_boost_classifier.fit(x_train, y_train)

mlp_classifier.fit(x_train, y_train)
gaussian_nb.fit(x_train, y_train)

voting_classifier_tree.fit(x_train, y_train)
voting_classifier_boost.fit(x_train, y_train)
voting_classifier_nb.fit(x_train, y_train)
voting_classifier_mix.fit(x_train, y_train)
voting_classifier_all.fit(x_train, y_train)


# See the scores
print(f'Logistic Regression Model Score: {logistic_regression.score(x_test, y_test)}')
print(f'Tree Model Score: {tree_classifier.score(x_test, y_test)}')
print(f'Extra Tree Model Score: {extra_tree_classifier.score(x_test, y_test)}')
print(f'AdaBoost Model Score: {adaboost_classifier.score(x_test, y_test)}')
print(f'Extra Trees Model Score: {extra_trees_classifier.score(x_test, y_test)}')
print(f'Bagging Model Score: {bagging_classifier.score(x_test, y_test)}')
print(f'Gradient Boost Model Score: {gradient_boost_classifier.score(x_test, y_test)}')
print(f'MultiLayer Perceptron Model Score: {mlp_classifier.score(x_test, y_test)}')
print(f'Gaussian Nayve Baes Model Score: {gaussian_nb.score(x_test, y_test)}')
print(f'VotingClassifier Tree Models Score: {voting_classifier_tree.score(x_test, y_test)}')
print(f'VotingClassifier Boost Models Score: {voting_classifier_boost.score(x_test, y_test)}')
print(f'VotingClassifier Naive Bayes Models Score: {voting_classifier_nb.score(x_test, y_test)}')
print(f'VotingClassifier Mixed Models Score: {voting_classifier_mix.score(x_test, y_test)}')
print(f'VotingClassifier All Models Score: {voting_classifier_all.score(x_test, y_test)}')


# Validate the scores
print(f'Logistic Regression Model Score: {logistic_regression.score(x_validation, y_validation)}')
print(f'Tree Model Score: {tree_classifier.score(x_validation, y_validation)}')
print(f'Extra Tree Model Score: {extra_tree_classifier.score(x_validation, y_validation)}')
print(f'AdaBoost Model Score: {adaboost_classifier.score(x_validation, y_validation)}')
print(f'Extra Trees Model Score: {extra_trees_classifier.score(x_validation, y_validation)}')
print(f'Bagging Model Score: {bagging_classifier.score(x_validation, y_validation)}')
print(f'Gradient Boost Model Score: {gradient_boost_classifier.score(x_validation, y_validation)}')
print(f'MultiLayer Perceptron Model Score: {mlp_classifier.score(x_validation, y_validation)}')
print(f'Gaussian Nayve Baes Model Score: {gaussian_nb.score(x_validation, y_validation)}')
print(f'VotingClassifier Tree Models Score: {voting_classifier_tree.score(x_validation, y_validation)}')
print(f'VotingClassifier Boost Models Score: {voting_classifier_boost.score(x_validation, y_validation)}')
print(f'VotingClassifier Naive Bayes Models Score: {voting_classifier_nb.score(x_validation, y_validation)}')
print(f'VotingClassifier Mixed Models Score: {voting_classifier_mix.score(x_validation, y_validation)}')
print(f'VotingClassifier All Models Score: {voting_classifier_all.score(x_validation, y_validation)}')
