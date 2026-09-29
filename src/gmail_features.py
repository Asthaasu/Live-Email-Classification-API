import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Project root = one folder above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Correct path based on your folder screenshot
DATASET_PATH = PROJECT_ROOT / "data" / "gmail_labeled_dataset.csv"


def extract_text_features(subject, sender, body=""):
    """Extract the required basic text features."""

    subject = str(subject) if pd.notna(subject) else ""
    sender = str(sender) if pd.notna(sender) else ""
    body = str(body) if pd.notna(body) else ""

    return {
        "subject_length": len(subject),
        "subject_word_count": len(subject.split()),
        "num_uppercase": sum(1 for char in subject if char.isupper()),
        "num_digits": sum(1 for char in subject if char.isdigit()),
        "num_special_chars": len(re.findall(r"[^\w\s@.-]", subject)),
        "has_urgent_keywords": int(
            any(
                keyword in subject.lower()
                for keyword in [
                    "urgent",
                    "click",
                    "verify",
                    "confirm",
                    "act now",
                    "limited time",
                ]
            )
        ),
        "num_urls": len(re.findall(r"(http|www)", subject.lower())),
        "sender_length": len(sender),
        "sender_has_plus": int("+" in sender),
    }


def extract_sender_features(sender):
    """Extract sender-domain features."""

    sender = str(sender) if pd.notna(sender) else ""

    if "@" not in sender:
        return {
            "is_free_email": 0,
            "domain_length": 0,
        }

    domain = sender.split("@")[-1].lower()

    free_domains = {
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
    }

    return {
        "is_free_email": int(domain in free_domains),
        "domain_length": len(domain),
    }


def extract_email_features(subject, sender, body=""):
    """Combine text and sender features for one email."""

    features = {}

    features.update(
        extract_text_features(subject, sender, body)
    )

    features.update(
        extract_sender_features(sender)
    )

    return features


def prepare_training_data(dataset_path=DATASET_PATH):
    """Load Gmail data and prepare X, y, and feature names."""

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at:\n{dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Total emails: {len(df)}")

    required_columns = ["subject", "sender", "body", "is_spam"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    extracted_features = df.apply(
        lambda row: extract_email_features(
            row["subject"],
            row["sender"],
            row["body"],
        ),
        axis=1,
    )

    features_df = pd.DataFrame(list(extracted_features))

    y = df["is_spam"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    return X_scaled, y, features_df.columns, scaler


if __name__ == "__main__":

    test_email = {
        "subject": "Urgent: Verify your account NOW!!!",
        "sender": "noreply+xyz@phishing-bank.com",
        "body": "",
    }

    print("\nTEST EMAIL FEATURES:\n")
    print(extract_email_features(**test_email))

    X, y, feature_names = prepare_training_data()

    print("\nFEATURE ENGINEERING COMPLETE")
    print(f"Training data shape: {X.shape}")
    print(f"Spam ratio: {y.mean():.2%}")

    print("\nFEATURE NAMES:")
    print(list(feature_names))
