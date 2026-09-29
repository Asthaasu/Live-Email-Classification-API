from gmail_auth import authenticate_gmail
from fetch_gmail_dataset import fetch_emails_optimized
from googleapiclient import discovery
import csv


def fetch_spam_folder_emails(creds, max_results=100):
    """Fetch emails from Gmail Spam folder"""
    service = discovery.build("gmail", "v1", credentials=creds)

    labels = service.users().labels().list(userId="me").execute()

    spam_label_id = next(
        (
            label["id"]
            for label in labels["labels"]
            if label["name"] == "[Gmail]/Spam"
        ),
        None
    )

    if not spam_label_id:
        print("Spam folder not found. Using query instead.")
        return fetch_emails_optimized(creds, max_results, query="is:spam")

    return fetch_emails_optimized(
        creds,
        max_results,
        query=f"label:{spam_label_id}"
    )


def label_emails_dataset(creds, output_file="../data/gmail_labeled_dataset.csv"):
    print("Fetching legitimate emails (inbox)...")
    legit = fetch_emails_optimized(
        creds,
        max_results=400,
        query="is:inbox"
    )

    for email in legit:
        email["is_spam"] = 0

    print("Fetching spam emails (spam folder)...")
    spam = fetch_spam_folder_emails(
        creds,
        max_results=100
    )

    for email in spam:
        email["is_spam"] = 1

    dataset = legit + spam

    print(f"Total emails: {len(dataset)}")
    print(f"Legitimate: {sum(1 for email in dataset if email['is_spam'] == 0)}")
    print(f"Spam: {sum(1 for email in dataset if email['is_spam'] == 1)}")

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["subject", "sender", "body", "is_spam"]
        )
        writer.writeheader()
        writer.writerows(dataset)

    print(f"\nDataset saved successfully: {output_file}")
    return dataset


if __name__ == "__main__":
    creds = authenticate_gmail()

    dataset = label_emails_dataset(
        creds,
        output_file="../data/gmail_labeled_dataset.csv"
    )
