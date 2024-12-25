# Xgboost algorithm
import xgboost as xgb
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# Create sample dataset
X, y = make_multilabel_classification(n_samples = 3000, n_features = 45,
                                      n_classes = 20, n_labels = 1, allow_unlabeled = False,
                                      random_state = 42)

# Split datasets into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 123)


# Create XGBoost instance with default hyper-parameters
xgb_estimator = xgb.XGBClassifier(objective= 'binary:logistic')

# Create MultiOutputClassifier instance with XGBoost model inside
multilabel_model = MultiOutputClassifier(xgb_estimator)

# fit the model
multilabel_model.fit(X_train, y_train)

# evaluate on the test data
print("Accuracy on the test data:{:.1f}%".format(accuracy_score(y_test,
                                                                multilabel_model.predict(X_test)*100)))
