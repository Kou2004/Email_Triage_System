import base64
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# Google Client Libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# 1. Page Configuration
st.set_page_config(page_title="Email Triage Assistant", layout="wide")
st.title("📧 Intelligent Live Email Triage Dashboard")
st.write("Fetch real-time data using the Gmail API, predict priorities, and draft replies.")

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# 2. Cache ML Artifacts for Speed
@st.cache_resource
def load_pipeline_assets():
    artifacts = joblib.load("email_triage_artifacts.joblib")
    return artifacts

try:
    assets = load_pipeline_assets()
    tfidf = assets["tfidf_vectorizer"]
    svd = assets["svd_transformer"]
    model = assets["model"]
    st.success("🎉 Machine learning assets successfully loaded!")
except Exception as e:
    st.error(f"❌ Error loading assets: {e}")
    st.stop()


# --- Helper Function: Gmail API Authentication ---
def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                st.error("❌ 'credentials.json' missing from workspace. Please add it to download live data.")
                st.stop()
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


# --- Helper Function: Extract and Clean Email Body Text ---
def get_mime_parts(payload):
    """Recursively traverses all nesting levels to pull out text/plain or text/html chunks."""
    parts_list = []

    if 'parts' in payload:
        for part in payload['parts']:
            parts_list.extend(get_mime_parts(part))
    elif payload.get('mimeType') in ['text/plain', 'text/html']:
        parts_list.append(payload)

    return parts_list


def parse_email_body(email_data):
    """
    Expects the full 'msg_detail' dictionary returned from the Gmail API.
    Processes headers, handles nested multipart bodies, strips out HTML code, and returns text.
    """
    headers = email_data.get('payload', {}).get('headers', [])
    payload = email_data.get('payload', {})

    sender = ""
    cc_count = 0

    # --- 1. CHECK SENDER AND CC COUNT FROM HEADERS ---
    for header in headers:
        name = header.get('name', '').lower()
        value = header.get('value', '')

        if name == 'from':
            sender = value.lower()
        elif name == 'cc':
            cc_list = [email.strip() for email in value.split(',') if email.strip()]
            cc_count = len(cc_list)

    # --- 2. IGNORE IITR SPAM NOTIFICATIONS ---
    if "IITR" in sender and "Spam" in sender:
        return None, 0

    # --- 3. EXTRACT EMAIL BODY USING RECURSION ---
    body = ""

    all_text_parts = get_mime_parts(payload)

    if all_text_parts:
        for part in all_text_parts:
            data = part.get('body', {}).get('data', '')
            if data:
                decoded_bytes = base64.urlsafe_b64decode(data)
                body += decoded_bytes.decode('utf-8', errors='ignore')
    else:
        data = payload.get('body', {}).get('data', '')
        if data:
            body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    if not body.strip():
        return "[No viewable body content found]", cc_count

    # --- 4. STRIP HTML AND CLEAN TEXT ---
    soup = BeautifulSoup(body, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    clean_text = soup.get_text(separator=" ")

    lines = (line.strip() for line in clean_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    final_text = "\n".join(chunk for chunk in chunks if chunk)

    final_text = final_text if final_text.strip() else "[No text content found]"

    return final_text, cc_count


# --- Helper Function: Gemini Answers Engine ---
from google import genai

def generate_reply(subject, body):
    try:
        client = genai.Client(api_key="your api_token")
        prompt = f"Write a brief, professional, 1-sentence draft reply to this email.\nSubject: {subject}\nBody: {body}"
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Automatic reply draft unavailable: {str(e)}"


# 3. Create Sidebar for System Configurations
st.sidebar.header("⚙️ Configuration Rules")
st.sidebar.info("Dashboard ranking updates dynamically using raw ML prediction weights.")


# 4. User Inputs Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Live Fetch Settings")
    days_back = st.slider("1. Lookback Window (Select days back to parse):", min_value=1, max_value=30, value=3)
    max_emails = st.number_input("2. Max emails to display (ranked by score):", min_value=1, max_value=50, value=1)

with col2:
    st.subheader("🧮 Custom Priority Score Information")
    st.info(
        """
    **Current Pipeline Framework:**
    * Priority score is pulled directly from the model probabilities.
    """
    )


# 5. Live Processing Execution Engine
if st.button("Fetch and Triage Live Gmail Traffic", type="primary"):
    with st.spinner("Connecting to Gmail API securely..."):
        try:
            service = get_gmail_service()
        except Exception as auth_err:
            st.error(f"Authentication Failed: {auth_err}")
            st.stop()

    query_string = f"newer_than:{days_back}d"

    with st.spinner(f"Searching mailbox for messages matching string: `{query_string}`"):
        results = service.users().messages().list(userId='me', q=query_string, maxResults=50).execute()
        messages = results.get('messages', [])

    if not messages:
        st.warning("No new emails found matching your filter criteria timeframe.")
        st.stop()

    st.success(f"Successfully downloaded metadata for {len(messages)} items. Running triage evaluation...")

    email_records = []
    urgent_words = [
        "dues", "students", "dear", "approved", "may", "roorkee", "marked", "open",
        "iit", "year", "academic", "office", "institute", "sports", "bhawan", "th",
        "posted", "notice", "email", "june", "greetings", "event", "national",
        "status", "changed", "committee", "reminder", "fwd", "pm", "session",
        "council", "inform", "affairs", "program", "kumar", "evaluation", "cdc",
        "regarding", "career", "awards", "please", "vande", "mataram", "club",
        "significance"
    ]

    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()

        payload = msg_detail.get('payload', {})
        headers = payload.get('headers', [])

        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "[No Subject]")
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown Sender")

        body, cc_count = parse_email_body(msg_detail)

        if body is None:
            continue

        combined_text = f"{subject} {body}"

        # --- ML Prediction Block ---
        text_tfidf = tfidf.transform([combined_text])
        text_svd = svd.transform(text_tfidf)

        svd_cols = [f"tf_idf_concept_{i}" for i in range(text_svd.shape[1])]
        X_text_features = pd.DataFrame(text_svd, columns=svd_cols)

        X_text_features['CC_count'] = cc_count
        X_text_features['contain_top_word'] = 1.0 if any(w in combined_text.lower() for w in urgent_words) else 0.0
        X_text_features['has_attachment'] = 1.0 if 'parts' in payload and len(payload['parts']) > 1 else 0.0
        X_text_features['sender_percentage'] = 0.05
        X_text_features['thread_count'] = 0

        expected_features = list(model.feature_names_in_)
        X_features = X_text_features.reindex(columns=expected_features, fill_value=0.0)

        p = float(model.predict_proba(X_features)[0, 1])

        email_records.append({
            'sender': sender,
            'subject': subject,
            'body': body,
            'priority_score': p,
            'has_attachment': X_text_features['has_attachment'].values[0],
            'CC_count': cc_count
        })

    # Sort results directly by the probability metric
    df_live = pd.DataFrame(email_records)
    df_results = df_live.sort_values(by='priority_score', ascending=False).head(max_emails)

    # 6. Display Completed Visual Output Cards
    st.markdown("---")
    st.subheader(f"📊 Live Priority Ranking Report (Top {len(df_results)} Items)")

    for idx, row in df_results.iterrows():
        sub_text = row['subject']
        body_text = row['body']
        sender_text = row['sender']
        score = row['priority_score']

        with st.expander(f"📩 [{score:.4f}] - From: {sender_text[:40]} | Subject: {sub_text[:40]}...", expanded=True):

            # Simplified Metadata Metrics Row Layout
            m1, m2 = st.columns(2)
            m1.metric("Sender Account Identifier", sender_text)
            m2.metric("Priority Score (Model Probability)", f"{score:.4f}")

            # Heuristic Rule Explanations Engine
            found_words = [w for w in urgent_words if w in sub_text.lower() or w in body_text.lower()]
            heuristics = []

            if found_words:
                heuristics.append(f"Contains Recent Words: `{', '.join(found_words)}`")
            if row['has_attachment'] == 1.0:
                heuristics.append("This email may have important attachment.")
            if row['CC_count'] > 1:
                heuristics.append('This email has many CC.')
            if not heuristics:
                heuristics.append("Prioritization calculated smoothly based entirely on latently computed model vectors.")

            st.write("---")
            st.markdown("**📌 Priority Insights & Triage Reasons:**")

            reasons_markdown = ""
            for rule in heuristics:
                reasons_markdown += f"* {rule}\n"

            st.info(reasons_markdown)

            # --- Custom Keyword Highlighting Logic ---
            highlighted_sub = sub_text
            highlighted_body = body_text

            for word in urgent_words:
                pattern = re.compile(rf'\b({word})\b', re.IGNORECASE)
                highlighted_sub = pattern.sub(
                    r"<mark style='background-color: #ffeb3b; color: black; font-weight: bold; padding: 2px;'>\1</mark>",
                    highlighted_sub
                )
                highlighted_body = pattern.sub(
                    r"<mark style='background-color: #ffeb3b; color: black; font-weight: bold; padding: 2px;'>\1</mark>",
                    highlighted_body
                )

            # Structured Data Presentation Blocks
            st.markdown("**Subject:**")
            st.markdown(highlighted_sub if highlighted_sub.strip() else "[No Subject]", unsafe_allow_html=True)

            st.markdown("**Message Snippet / Content Body:**")
            safe_body = highlighted_body if highlighted_body.strip() else "[No body content found]"
            st.markdown(
                f"<div style='background-color: #1e1e1e; padding: 12px; border-radius: 5px; max-height: 200px; overflow-y: auto; color: white;'>{safe_body}</div>",
                unsafe_allow_html=True
            )

            # Gemini Response Field Box Layout
            st.markdown("<br>**🤖 Gemini Intelligent Response Suggestion Draft:**", unsafe_allow_html=True)
            with st.spinner("Drafting live reply..."):
                reply_text = generate_reply(sub_text, body_text)
                st.code(reply_text, language="markdown")