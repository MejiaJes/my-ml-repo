import sys
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ACCURACY_THRESHOLD = 0.90

def evaluate_model():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Threshold: {ACCURACY_THRESHOLD}")

    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}")
        sys.exit(1)

    print("PASSED: model meets quality threshold")
    sys.exit(0)

if __name__ == "__main__":
    evaluate_model()
