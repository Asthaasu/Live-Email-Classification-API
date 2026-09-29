from gmail_auth import authenticate_gmail
from googleapiclient.discovery import build

creds = authenticate_gmail()

service = build(
    "gmail",
    "v1",
    credentials=creds
)

results = service.users().messages().list(
    userId="me",
    maxResults=10
).execute()

messages = results.get("messages", [])

print(f"\nFound {len(messages)} emails\n")

for i, msg in enumerate(messages):

    full_msg = service.users().messages().get(
        userId="me",
        id=msg["id"]
    ).execute()

    headers = full_msg["payload"].get("headers", [])

    subject = "No Subject"
    sender = "Unknown Sender"

    for h in headers:
        if h["name"] == "Subject":
            subject = h["value"]

        if h["name"] == "From":
            sender = h["value"]

    snippet = full_msg.get("snippet", "")

    print("=" * 60)
    print(f"Email #{i+1}")
    print(f"From: {sender}")
    print(f"Subject: {subject}")
    print(f"Snippet: {snippet}")
    print("=" * 60)
