import pandas as pd
import random as rn
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
from sklearn.model_selection import cross_val_score
import warnings


RANDOM_STATE = 30


# Load dataset from CSV
df = pd.read_csv('../clean_dataset.csv').sample(frac=.5)

# Predefine some states
warnings.simplefilter(action='ignore')
rn.seed(RANDOM_STATE)

# Get the train and test data
train_df = df.sample(frac=.8, random_state=RANDOM_STATE)
test_df = df.sample(frac=.2, random_state=RANDOM_STATE)

x_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]

x_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]


# Instantiate models
# logistic_regression = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#
# tree_classifier = DecisionTreeClassifier()
# extra_tree_classifier = ExtraTreeClassifier()
# tree_estimators = [('tc', DecisionTreeClassifier()), ('etc', ExtraTreeClassifier())]
# voting_classifier_tree = VotingClassifier(estimators=tree_estimators)
#
adaboost_classifier = AdaBoostClassifier()
# extra_trees_classifier = ExtraTreesClassifier(n_estimators=70)
# bagging_classifier = BaggingClassifier()
# random_forest_classifier = RandomForestClassifier()
# gradient_boost_classifier = GradientBoostingClassifier()
# boost_estimators = [
#     ('adac', AdaBoostClassifier()), ('etsc', ExtraTreesClassifier(n_estimators=70)),
#     ('bc', BaggingClassifier()), ('rfc', RandomForestClassifier()), ('gbc', GradientBoostingClassifier())]
# voting_classifier_boost = VotingClassifier(estimators=boost_estimators, voting='soft')
#
# mlp_classifier = MLPClassifier(max_iter=700, random_state=RANDOM_STATE)
# gaussian_nb = GaussianNB()
# nb_estimators = [('mc', MLPClassifier()), ('gnb', GaussianNB())]
# voting_classifier_nb = VotingClassifier(estimators=nb_estimators)
#
# mix_estimators = [
#     ('le', LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')),
#     ('te', rn.choice(tree_estimators)[1]), *rn.sample(boost_estimators, 4), *rn.sample(nb_estimators, 2)]
# voting_classifier_mix = VotingClassifier(estimators=mix_estimators)
#
#
# all_estimators = [('lgr',  LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')),
#                   *tree_estimators, *boost_estimators, *nb_estimators]
# voting_classifier_all = VotingClassifier(estimators=all_estimators)
#
#
# Train all models, (it will take some time)
# logistic_regression.fit(x_train, y_train)
#
# tree_classifier.fit(x_train, y_train)
# extra_tree_classifier.fit(x_train, y_train)
#
adaboost_classifier.fit(x_train, y_train)
# extra_trees_classifier.fit(x_train, y_train)
# bagging_classifier.fit(x_train, y_train)
# gradient_boost_classifier.fit(x_train, y_train)
#
# mlp_classifier.fit(x_train, y_train)
# gaussian_nb.fit(x_train, y_train)
#
# voting_classifier_tree.fit(x_train, y_train)
# voting_classifier_boost.fit(x_train, y_train)
# voting_classifier_nb.fit(x_train, y_train)
# voting_classifier_mix.fit(x_train, y_train)
# voting_classifier_all.fit(x_train, y_train)
#
#
# See the scores
# print(f'Logistic Regression Model Score: {logistic_regression.score(x_test, y_test)}')
# print(f'Decision Tree Model Score: {tree_classifier.score(x_test, y_test)}')
# print(f'Extra Tree Model Score: {extra_tree_classifier.score(x_test, y_test)}')
print(f'AdaBoost Model Score: {adaboost_classifier.score(x_test, y_test)}')
# print(f'Extra Trees Model Score: {extra_trees_classifier.score(x_test, y_test)}')
# print(f'Bagging Model Score: {bagging_classifier.score(x_test, y_test)}')
# print(f'Gradient Boost Model Score: {gradient_boost_classifier.score(x_test, y_test)}')
# print(f'MultiLayer Perceptron Model Score: {mlp_classifier.score(x_test, y_test)}')
# print(f'Gaussian Nayve Baes Model Score: {gaussian_nb.score(x_test, y_test)}')
# print(f'VotingClassifier Tree Models Score: {voting_classifier_tree.score(x_test, y_test)}')
# print(f'VotingClassifier Boost Models Score: {voting_classifier_boost.score(x_test, y_test)}')
# print(f'VotingClassifier Naive Bayes Models Score: {voting_classifier_nb.score(x_test, y_test)}')
# print(f'VotingClassifier Mixed Models Score: {voting_classifier_mix.score(x_test, y_test)}')
# print(f'VotingClassifier All Models Score: {voting_classifier_all.score(x_test, y_test)}')
#
#
print('\nLogistic Regression cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(LogisticRegression(solver='lbfgs', multi_class='multinomial'), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(LogisticRegression(solver='lbfgs', multi_class='multinomial'), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(LogisticRegression(solver='lbfgs', multi_class='multinomial'), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(LogisticRegression(solver='lbfgs', multi_class='multinomial'), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nDecision Tree cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(DecisionTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(DecisionTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(DecisionTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(DecisionTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nExtra Tree cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(ExtraTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(ExtraTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(ExtraTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(ExtraTreeClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nVoting Tree cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(VotingClassifier(estimators=tree_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(VotingClassifier(estimators=tree_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(VotingClassifier(estimators=tree_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(VotingClassifier(estimators=tree_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nAdaBoost cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(AdaBoostClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(AdaBoostClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(AdaBoostClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(AdaBoostClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nExtra Trees cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(ExtraTreesClassifier(n_estimators=70), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(ExtraTreesClassifier(n_estimators=70), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(ExtraTreesClassifier(n_estimators=70), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(ExtraTreesClassifier(n_estimators=70), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nBagging cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(BaggingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(BaggingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(BaggingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(BaggingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro '))/10) * 100, 2)}%")


print('\nRandom Forest cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(RandomForestClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(RandomForestClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(RandomForestClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(RandomForestClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nGradient cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(GradientBoostingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(GradientBoostingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(GradientBoostingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(GradientBoostingClassifier(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nVoting Boost cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(VotingClassifier(estimators=boost_estimators, voting='soft'), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(VotingClassifier(estimators=boost_estimators, voting='soft'), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(VotingClassifier(estimators=boost_estimators, voting='soft'), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(VotingClassifier(estimators=boost_estimators, voting='soft'), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nMLP cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(MLPClassifier(max_iter=700, random_state=RANDOM_STATE), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(MLPClassifier(max_iter=700, random_state=RANDOM_STATE), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(MLPClassifier(max_iter=700, random_state=RANDOM_STATE), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(MLPClassifier(max_iter=700, random_state=RANDOM_STATE), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nGaussian cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(GaussianNB(), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(GaussianNB(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(GaussianNB(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(GaussianNB(), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nNaiveBayes Voting cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(VotingClassifier(estimators=nb_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(VotingClassifier(estimators=nb_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(VotingClassifier(estimators=nb_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(VotingClassifier(estimators=nb_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nMixed Voting cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(VotingClassifier(estimators=mix_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(VotingClassifier(estimators=mix_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(VotingClassifier(estimators=mix_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(VotingClassifier(estimators=mix_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")


print('\nAll Voting cross-validation F1, Precision and Recall')
print(f"Accuracy  : {round((sum(cross_val_score(VotingClassifier(estimators=all_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10))/10) * 100, 2)}%")
print(f"F1        : {round((sum(cross_val_score(VotingClassifier(estimators=all_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='f1_macro'))/10) * 100, 2)}%")
print(f"Precision : {round((sum(cross_val_score(VotingClassifier(estimators=all_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='precision_macro'))/10) * 100, 2)}%")
print(f"Recall    : {round((sum(cross_val_score(VotingClassifier(estimators=all_estimators), df.iloc[:, 1:], df.iloc[:, 0], cv=10, scoring='recall_macro'))/10) * 100, 2)}%")
