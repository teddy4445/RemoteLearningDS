# library imports
import os
import math
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

# project imports


class RandomForest:

    # global #
    ERROR_VAL = -1
    # end - global #

    # consts #
    random_forst = 1
    single_tree = 2
    # end - consts #

    def __init__(self):
        self.model = None
        self.type = None

    def get_model(self) -> RandomForestClassifier:
        return self.model

    def set_model(self,
                  model: RandomForestClassifier) -> None:
        self.model = model

    def set_type(self,
                 type: int) -> None:
        self.type = type

    def train(self,
              x: list,
              y: list,
              type: int = 1,
              max_depth: int = 2,
              n_estimators: int = 10,
              random_state: int = 0,
              criterion: str = "entropy") -> None:
        if type == RandomForest.random_forst:
            clf = RandomForestClassifier(criterion=criterion,
                                         max_depth=max_depth,
                                         random_state=random_state,
                                         n_estimators=n_estimators)
        else:
            clf = tree.DecisionTreeClassifier(criterion=criterion,
                                              max_depth=max_depth,
                                              random_state=random_state)
        clf.fit(x, y)
        self.model = clf
        self.type = type

    def export_graph(self,
                     feature_names: list,
                     class_names: list,
                     print_text: str,
                     print_to_console: bool = False):
        if self.type == RandomForest.random_forst:
            for i in range(len(self.model.model.estimators_)):
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
                tree.plot_tree(self.model.model.estimators_[i],
                               feature_names=feature_names,
                               impurity=True,
                               proportion=True,
                               class_names=class_names,
                               filled=False)
                fig.savefig('save_name_rf_tree_answer_{}.png'.format(print_text, i+1))
                if print_to_console:
                    print("\nRandom Forest (Tree #{}): \n{}".format(i, export_text(decision_tree=self.model.model.estimators_[i])))
                    print("Feature Importance: {}\n\n".format(", ".join(["{:.0f}%".format(math.floor(100 * val)) for val in self.model.model.estimators_[i].feature_importances_])))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
            tree.plot_tree(self.model.model,
                           feature_names=feature_names,
                           impurity=True,
                           proportion=True,
                           class_names=class_names,
                           filled=False)
            fig.savefig('{}_dtree_answer.png'.format(print_text))
            if print_to_console:
                print("\nDecision Tree: \n{}".format(export_text(decision_tree=self.model.model)))
                print("Feature Importance: {}\n\n".format(", ".join(["{:.0f}%".format(math.floor(100 * val)) for val in self.model.model.feature_importances_])))
        plt.close()

    def print_importance(self,
                         name: str,
                         ordered: bool = False):
        if self.type == RandomForest.single_tree:
            importances = [val * 100 for val in self.model.model.feature_importances_]
        else:
            importances = []
            for i in range(len(self.model.model.estimators_)):
                if len(importances) == 0:
                    importances = [val for val in self.model.model.estimators_[i].feature_importances_]
                else:
                    for val in self.model.model.estimators_[i].feature_importances_:
                        importances += val
            importances = [val / len(self.model.model.estimators_) for val in importances]
            summer = sum(importances)
            importances = [val / summer * 100 for val in importances]

        if ordered:
            importances = [(i, val) for i, val in enumerate(importances)]
            importances = sorted(importances, key=lambda x: x[1])
            indeces = [item[0] for item in importances]
            importances = [item[1] for item in importances]
        else:
            indeces = [i for i in range(len(importances))]
        std = np.std(self.model.model.feature_importances_, axis=0)
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importance")
        plt.xlabel("Feature index")
        plt.ylabel("Feature's importance (percent)")
        plt.bar(range(len(importances)),
                importances,
                color="blue",
                yerr=std,
                align="center")
        print(importances)
        plt.xticks(range(len(importances)), indeces)
        plt.xlim([-1, len(importances)])
        plt.ylim([0, 100])
        plt.savefig("{}_importance_graph.png".format(name))
        plt.close()

    def test(self,
             x_test: list,
             y_test: list):
        return self.model.score(x_test, y_test) if len(x_test) > 0 and len(y_test) > 0 else RandomForest.ERROR_VAL

    @staticmethod
    def load(model: RandomForestClassifier):
        rf = RandomForest()
        rf.set_model(model=model)
        return rf

    def predict(self,
                x: list):
        if self.model:
            return self.model.predict([x])
        else:
            raise Exception("Model is not ready")

    def predict_list(self,
                     x_list: list):
        if self.model:
            return self.model.predict(x_list)
        else:
            raise Exception("Model is not ready")

    def explain_decision(self,
                         sample: list) -> None:
        feature = self.model.model.tree_.feature
        threshold = self.model.model.tree_.threshold
        node_indicator = self.model.model.decision_path([sample])
        leave_id = self.model.model.apply([sample])

        sample_id = 0
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        print('Rules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue

            if sample[feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision id node %s : (sample[%s] (= %s) %s %s)"
                  % (node_id,
                     feature[node_id],
                     sample[feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))
