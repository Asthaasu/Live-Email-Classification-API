import os
import pickle
import pandas as pd

from gmail_features import extract_features

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "gmail_random_forest.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "gmail_feature_names.pkl"
)


def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(FEATURE_NAMES_PATH, "rb") as file:
        feature_names = pickle.load(file)

    return model, feature_names


def predict_email(subject, sender, snippet):
    model, feature_names = load_model()

    features = extract_features(subject, sender, snippet)

    # Keep exactly the same feature order used during training
    input_df = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]
    spam_probability = model.predict_proba(input_df)[0][1]

    print("\n" + "=" * 50)
    print("GMAIL SPAM PREDICTION")
    print("=" * 50)
    print(f"Subject: {subject}")
    print(f"Sender: {sender}")
    print(f"\nPrediction: {'SPAM' if prediction == 1 else 'LEGITIMATE'}")
    print(f"Spam probability: {spam_probability:.2%}")


if __name__ == "__main__":
    test_subject = "URGENT: Verify your account now!!!"
    test_sender = "security-alert@unknown-bank-login.com"
    test_snippet = (
        "Your account will be suspended today. "
        "Click the link immediately to verify your details."
    )

    predict_email(
        subject=test_subject,
        sender=test_sender,
        snippet=test_snippet
    )
