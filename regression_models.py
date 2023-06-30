import numpy as np
import pandas as pd
import os
import copy
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle


class BoostSHr(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, n_iter=10, fi=0.05, learning_rate=1.):
        """
            Boost SH : Boosting classification algorithm for multiview with shared weights
            Greedy approach in which each view is tested to evaluate the one with larger
            edge

            Arguments:
                base_estimator {sklearn model} -- Base classifier to use on each view
                n_iter {int} -- Number of boosting iterations
                learning_rate {float} --  Learning rate for boosting (default: 1)
        """
        super(BoostSHr, self).__init__()
        self.base_estimator = base_estimator
        self.views = {}

        self.models = []
        self.classes = []
        self.alphas = []
        self.views_selected = []
        self.weights = []
        self.eps = 10 ** (-6)

        self.n_iter = n_iter
        self.fi = fi
        self.learning_rate = learning_rate

    def fit(self, X, y, forecast_cv=None, sample_weights=None):
        """
            Fit the model by adding models in an adaboost fashion

            Arguments:
                X {Dict of pd Dataframe} -- Views to use for the task
                y {pd Dataframe} -- Labels - Index has to be contained in view union
                forecast_cv {int} -- Number of fold used to estimate the edge
                    (default: None - Performance are computed on training set)
        """
        self.check_input(X, y)

        self.views = copy.deepcopy(X)
        self.classes = np.unique(y)

        index = self.__index_union__(self.views)

        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        for t in range(self.n_iter):

            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            selected = {'max_edge': -np.inf}
            for v in self.views:
                mask = self.views[v].index.tolist()
                weak_forecast = self.__compute_weak_forecast__(self.views[v], y[mask], self.weights[mask], forecast_cv, True)
                tmp = (weak_forecast['forecast'] - y[mask].values) / y[mask].values
                edge = 0
                if tmp < self.fi:
                    edge += (self.weights[mask].values * tmp).sum()

                if edge > selected['max_edge']:
                    selected = {'max_edge': edge, 'view': v, 'forecast': weak_forecast['forecast'], 'mask': mask,
                                'model': weak_forecast['model'], 'tmp': tmp}

            if (1 - selected['max_edge']) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            else:
                alpha = self.learning_rate * .5 * np.log((1 + selected['max_edge']) / (1 - selected['max_edge']))

            # Update weights
            self.weights[selected['mask']] *= np.exp(- alpha * selected['tmp'])

            self.models.append(selected['model'])
            self.alphas.append(alpha)
            self.views_selected.append(selected['view'])

            print('Iteration ', t)
            print('Winning view ', selected['view'])
            print('Edge ', selected['max_edge'])
            print('')

        return

    def initialize_weights(self, sample_weights=None, index=None):
        if sample_weights is None:
            # Initialize uniform distribution over weights
            self.weights = pd.Series(1, index=index)

        else:
            # Assign pre-defined distribution over weights
            self.weights = pd.Series(sample_weights, index=index)

    def __compute_weak_forecast__(self, data, labels, weights, forecast_cv=None, return_model=False):
        weak_forecast = dict()
        model = clone(self.base_estimator)

        data, labels, weights = shuffle(data, labels, weights)
        if forecast_cv is None:
            model.fit(data.values, labels.values, sample_weight=weights.values)
            forecast = model.predict(data.values)
        else:
            forecast = cross_val_predict(model, data.values, labels.values, cv=forecast_cv,
                                         fit_params={'sample_weight': weights.values})

        weak_forecast['forecast'] = forecast
        if return_model:
            weak_forecast['model'] = model

        return weak_forecast

    def predict_proba(self, X):
        index = self.__index_union__(X)
        predictions = np.zeros((len(index)))
        for t in range(len(self.models)):
            if self.views_selected[t] in X.keys():
                X_test = X[self.views_selected[t]]
                int_index = [list(index).index(i) for i in X_test.index]
                predictions[int_index] = self.models[t].predict(X_test.values)

        predictions = pd.DataFrame(predictions, index=index)

        return predictions

    def predict(self, X):
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        pp = self.predict_proba(X).idxmax(axis=1)
        return pp

    def check_input(self, X, y):
        self.check_X(X)
        self.check_y(y)

    def check_X(self, X):
        assert isinstance(X, dict), "Not right format for X"
        for key in X.keys():
            assert isinstance(X[key], pd.DataFrame)
            assert not X[key].empty, "Empty dataframe"

    def check_y(self, y):
        assert isinstance(y, pd.Series), "Not right format for y"
        # assert len(y.unique()) > 1, "One class in data"
        assert not y.empty, "Empty dataframe"

    def __index_union__(self, views):
        self.check_X(views)

        index = set([])
        for view in views:
            index = index.union(set(views[view].index))

        return list(index)


class IRBoostSHr(BoostSHr):

    def __init__(self, base_estimator, n_iter=10, learning_rate=1., sigma=0.15, gamma=0.3):
        """
            rBoost SH : Boosting classification for multiview with shared weights.
            Multi-arm bandit approach in which a view is selected at each iteration

            Arguments:
                base_estimator {sklearn model} -- Base classifier to use on each view
                n_iter {int} -- Number of boosting iterations
                learning_rate {float} -- Learning rate for boosting (default: 1)
        """
        super(IRBoostSHr, self).__init__(base_estimator, n_iter, learning_rate)
        self.sigma = sigma
        self.gamma = gamma

    def fit(self, X, y, forecast_cv=None, sample_weights=None):
        """
            Fit the model by adding models in a boosting fashion

            Arguments:
                X {Dict of pandas Dataframes} -- Views to use for the task
                y {pandas Series} -- Labels - Index has to be contained in view union
                forecast_cv {int} -- Number of fold used to estimate the edge
                    (default: None - Performance are computed on training set)
        """
        self.check_input(X, y)

        self.views = copy.deepcopy(X)
        self.classes = np.unique(y)
        K = len(self.views)
        possible_views = list(self.views.keys())

        index = self.__index_union__(self.views)
        # Reorder labels
        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        p_views = pd.Series(np.exp(self.sigma * self.gamma / 3 * np.sqrt(self.n_iter / K)), index=possible_views)

        for t in range(self.n_iter):

            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            # Bandit selection of best view
            q_views = (1 - self.gamma) * p_views / p_views.sum() + self.gamma / K
            selected_view = np.random.choice(possible_views, p=q_views)

            mask = self.views[selected_view].index.tolist()
            weak_forecast = self.__compute_weak_forecast__(self.views[selected_view], y[mask], self.weights[mask],
                                                           forecast_cv, True)
            # Calculate edge

            tmp = (weak_forecast['forecast'] - y[mask].values) / y[mask].values
            edge = (self.weights[mask].values * tmp).sum()

            if (1 - edge) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            elif edge <= 0:
                alpha = 0
            else:
                alpha = self.learning_rate * .5 * np.log((1 + edge) / (1 - edge))

            # Update weights
            self.weights[mask] *= np.exp(- alpha * tmp)

            # Update arm probability
            r_views = pd.Series(0, index=possible_views)
            square = np.sqrt(1 - edge ** 2) if edge < 1 else 0
            r_views[selected_view] = (1 - square) / q_views[selected_view]
            p_views *= np.exp(self.gamma / (3 * K) * (r_views + self.sigma / (q_views * np.sqrt(self.n_iter * K))))

            self.models.append(weak_forecast['model'])
            self.alphas.append(alpha)
            self.views_selected.append(selected_view)

            print('')
            print('Iteration ', t)
            print('Winning view ', selected_view)
            print('Edge ', edge)

        return

    def view_weights(self):
        """
            Return relative importance of the different views in the final decision
        """
        assert len(self.models) > 0, 'Model not trained'
        view_weights = pd.DataFrame({"view": self.views_selected, "alpha": np.abs(self.alphas)})
        return (view_weights.groupby('view').sum() / np.sum(np.abs(self.alphas))).sort_values('alpha')

    def save_views(self, path):
        df = pd.DataFrame(self.views_selected)
        if path.endswith('xlsx'):
            file = path
        else:
            file = os.path.join(path, 'winning views.xlsx')
        df.to_excel(file)


