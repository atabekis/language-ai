"""Testing"""
from methods.process import build_pipeline
from methods.reader import Reader
from config import __DATA_PATH__
from util import log

from sklearn.model_selection import GridSearchCV


class Tuner:
    """This class is used to tne the models found in the process.py file"""

    def __init__(self):
        reader = Reader(__DATA_PATH__,
                        clean=True,
                        split=True,
                        show_info=False)

        self.posts = reader.posts
        self.labels = reader.labels

        self.X_train, self.y_train = reader.train[0], reader.train[1]
        self.X_test, self.y_test = reader.test[0], reader.test[1]

        self.resampling_method = 'random_under'
        self.scoring = 'f1'

    def _grid_search_tuner(self, model_name: str, param_grid: dict[any], cv: int = 5):
        """Builds the pipeline and performs 5-fold cross validation to find the best model parameters
        :param model_name: str
            name of the model from: naive-bayes, svm, random-forest, logistic
        :param param_grid:
            pre-fitted parameters for the model
        :param cv:
            k value for the k-fold cross validation
        """
        pipeline = build_pipeline(model=model_name, resampling_method=self.resampling_method, verbose=True)

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=self.scoring,
                                   n_jobs=3, verbose=True)
        grid_search.fit(self.X_train, self.y_train)

        print(f'Best score: {grid_search.best_score_:%0.3f}')
        print('Best parameters set:')
        best_params = grid_search.best_params_
        for param in sorted(param_grid.keys()):
            print(f"\t{param}: {best_params[param]!r}")
        return best_params

    def tune_rf(self):
        """Estimate the parameters for the Random Forest classifier"""
        log('[Tuner] Tuning: random-forest model')
        model_name = 'random-forest'
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]}

        return self._grid_search_tuner(model_name=model_name, param_grid=param_grid)

    def tune_svm(self):
        """Estimate the parameters for the LinearSVC classifier"""

        log('[Tuner] Tuning: svm model')

        model_name = 'svm'
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__loss': ['hinge', 'squared_hinge'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__dual': [True, False]
        }
        return self._grid_search_tuner(model_name=model_name, param_grid=param_grid)


if __name__ == '__main__':
    tuner = Tuner()
    tuned_rf = tuner.tune_svm()
    print(tuned_rf)
