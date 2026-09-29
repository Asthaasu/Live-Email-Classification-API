from gmail_auth import authenticate_gmail
from googleapiclient.discovery import build
import pandas as pd


def fetch_emails_optimized(creds, max_results=100, query=""):
    """
    Fetch Gmail emails using a Gmail query.
    Returns: subject, sender, body
    """

    service = build("gmail", "v1", credentials=creds)

    try:
        results = service.users().messages().list(
            userId="me",
            maxResults=max_results,
            q=query
        ).execute()

        messages = results.get("messages", [])
        email_data = []

        print(f"Found {len(messages)} emails. Fetching metadata...")

        for msg in messages:
            try:
                full_msg = service.users().messages().get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From"]
                ).execute()

                headers = full_msg.get("payload", {}).get("headers", [])

                subject = next(
                    (
                        h["value"]
                        for h in headers
                        if h["name"].lower() == "subject"
                    ),
                    "[No Subject]"
                )

                sender = next(
                    (
                        h["value"]
                        for h in headers
                        if h["name"].lower() == "from"
                    ),
                    "[Unknown Sender]"
                )

                email_data.append({
                    "subject": subject,
                    "sender": sender,
                    "body": full_msg.get("snippet", "")
                })

            except Exception as error:
                print(f"Skipped one email: {error}")

        print(f"Successfully fetched {len(email_data)} emails.")
        return email_data

    except Exception as error:
        print(f"Gmail API error: {error}")
        return []


if __name__ == "__main__":
    creds = authenticate_gmail()

    emails = fetch_emails_optimized(
        creds,
        max_results=500,
        query=""
    )

    df = pd.DataFrame(emails)

    df.to_csv(
        "../data/gmail_dataset.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nSaved to ../data/gmail_dataset.csv")
    print(f"Total rows saved: {len(df)}")
