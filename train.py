from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelTrainer:
    # Registry beberapa model + logika baseline & tuning.
    # Tiap model dibungkus Pipeline([preprocessing, model]) supaya preprocessing
    # ikut di-fit pada data latih saja (anti leakage).

    def __init__(self, transformer, random_state=42):
        self.transformer = transformer
        self.random_state = random_state

        # nama -> (estimator default, distribusi hyperparameter untuk tuning)
        self.registry = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, class_weight="balanced"),
                {
                    "model__C": [0.01, 0.1, 1.0, 10.0],
                    "model__solver": ["lbfgs", "saga"],
                },
            ),
            "Decision Tree": (
                DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
                {
                    "model__max_depth": [None, 10, 20, 30],
                    "model__min_samples_leaf": [1, 2, 5, 10],
                    "model__criterion": ["gini", "entropy"],
                },
            ),
            "Random Forest": (
                RandomForestClassifier(class_weight="balanced",
                                       random_state=random_state, n_jobs=-1),
                {
                    "model__n_estimators": [200, 300, 400],
                    "model__max_depth": [None, 15, 25, 35],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2"],
                },
            ),
            "XGBoost": (
                XGBClassifier(random_state=random_state, eval_metric="mlogloss", n_jobs=-1),
                {
                    "model__n_estimators": [200, 300, 400],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.05, 0.1, 0.2],
                    "model__subsample": [0.8, 1.0],
                },
            ),
        }

    def _make_pipe(self, estimator):
        return Pipeline([("prep", self.transformer), ("model", estimator)])

    @staticmethod
    def _grid_size(param_dist):
        size = 1
        for v in param_dist.values():
            size *= len(v)
        return size

    def baseline_fit(self, name, X_tr, y_tr):
        # Latih satu model dengan parameter default.
        estimator, _ = self.registry[name]
        pipe = self._make_pipe(estimator)
        pipe.fit(X_tr, y_tr)
        return pipe

    def tune(self, name, X_tr, y_tr, n_iter=8, cv=3):
        # RandomizedSearchCV untuk satu model, dioptimalkan ke Macro F1.
        estimator, param_dist = self.registry[name]
        pipe = self._make_pipe(estimator)
        n_iter = min(n_iter, self._grid_size(param_dist))
        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=n_iter, scoring="f1_macro",
            cv=cv, random_state=self.random_state, n_jobs=-1, verbose=0,
        )
        search.fit(X_tr, y_tr)
        return search
