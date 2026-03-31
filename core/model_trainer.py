import optuna
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def tune_xgboost(self, X, y):
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
            model.fit(X, y)
            return model.score(X, y)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        return study.best_params

    def build_stack(self, X, y, best_params):
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(**best_params))
        ]
        # Meta-Learner is Logistic Regression
        stack = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5
        )
        stack.fit(X, y)
        joblib.dump(stack, 'models/stack_model.pkl')
        return stack