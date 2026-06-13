import os
import base64
from datetime import datetime
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def parse_sender(from_header):
    if '<' in from_header and '>' in from_header:
        name, email = from_header.split('<', 1)
        return name.strip().strip('"'), email.replace('>', '').strip()
    return "", from_header.strip()

def count_cc_recipients(cc_header):
    if not cc_header:
        return 0
    recipients = [r for r in cc_header.split(',') if r.strip()]
    return len(recipients)

def format_internal_date(timestamp_ms):
    try:
        return datetime.fromtimestamp(int(timestamp_ms) / 1000.0)
    except:
        return None

def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Extracted Emails', index=False)
        worksheet = writer.sheets['Extracted Emails']
        worksheet.views.sheetView[0].showGridLines = True
        worksheet.freeze_panes = "A2"
        
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            if col[0].value in ["Snippet", "Subject"]:
                worksheet.column_dimensions[col_letter].width = 40
            elif col[0].value in ["To", "BCC"]:
                worksheet.column_dimensions[col_letter].width = 30
            elif col[0].value in ["Labels"]:
                worksheet.column_dimensions[col_letter].width = 25
            elif col[0].value in ["Action Type Status", "Action System Time"]:
                worksheet.column_dimensions[col_letter].width = 22
            else:
                worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

def extract_all_emails():
    try:
        service = get_gmail_service()
        
        target_emails = 3000  # Cap at last 3000 emails
        messages = []
        next_page_token = None
        output_file = "Gmail_Extracted_Data.xlsx"
        
        print(f"🔍 Scanning your inbox to collect the latest {target_emails} message IDs...")
        
        while len(messages) < target_emails:
            batch_size = min(500, target_emails - len(messages))
            
            results = service.users().messages().list(
                userId='me', 
                maxResults=batch_size, 
                pageToken=next_page_token
            ).execute()
            
            batch_messages = results.get('messages', [])
            if not batch_messages:
                break
                
            messages.extend(batch_messages)
            next_page_token = results.get('nextPageToken')
            
            print(f"  -> Collected {len(messages)} message IDs...")
            
            if not next_page_token:
                break 

        if not messages:
            print("❌ No messages found.")
            return

        all_email_data = []
        print(f"\n🚀 Fetching details for {len(messages)} emails...")

        for idx, message in enumerate(messages, 1):
            try:
                msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
                
                msg_id = msg.get('id')
                thread_id = msg.get('threadId')
                snippet = msg.get('snippet', '')
                
                label_list = msg.get('labelIds', [])
                labels_str = ", ".join(label_list)
                
                # Deduce State Action
                action_type = "Read (Opened)"
                if "UNREAD" in label_list:
                    action_type = "Unread (No Action)"
                if "TRASH" in label_list:
                    action_type = "Deleted (Trashed)"
                elif "INBOX" not in label_list and "SENT" not in label_list:
                    action_type = "Archived"
                if "STARRED" in label_list:
                    action_type += " + Starred"

                internal_timestamp = msg.get('internalDate')
                action_dt = format_internal_date(internal_timestamp)
                action_time_str = action_dt.strftime('%Y-%m-%d %H:%M:%S') if action_dt else ""
                
                headers = msg['payload']['headers']
                subject = "No Subject"
                raw_from = "Unknown"
                to_field = "Unknown"
                cc_field = ""    
                bcc_field = ""   
                date_field = ""
                
                for header in headers:
                    name = header['name'].lower()
                    if name == 'subject':
                        subject = header['value']
                    elif name == 'from':
                        raw_from = header['value']
                    elif name == 'to':
                        to_field = header['value']
                    elif name == 'cc':
                        cc_field = header['value']
                    elif name == 'bcc':
                        bcc_field = header['value']
                    elif name == 'date':
                        date_field = header['value']

                sender_name, sender_email = parse_sender(raw_from)
                cc_count = count_cc_recipients(cc_field)
                
                payload = msg.get('payload', {})
                parts = payload.get('parts', [])
                has_attachments = "No"
                body_type = "Plain Text"
                
                if parts:
                    for part in parts:
                        if part.get('filename'):
                            has_attachments = "Yes"
                        if part.get('mimeType') == 'text/html':
                            body_type = "HTML"
                else:
                    if payload.get('mimeType') == 'text/html':
                        body_type = "HTML"

                all_email_data.append({
                    "Message ID": msg_id,
                    "Thread ID": thread_id,
                    "Email Date Header": date_field,
                    "Action System Time": action_time_str,     
                    "Action Type Status": action_type,     
                    "From (Name)": sender_name,
                    "From (Email)": sender_email,
                    "To": to_field,
                    "CC Count": cc_count,
                    "BCC": bcc_field,
                    "Subject": subject,
                    "Labels": labels_str,
                    "Snippet": snippet,
                    "Body Type": body_type,
                    "Has Attachments": has_attachments
                })
            except Exception as e:
                print(f"⚠️ Skipped email index {idx} due to structural error: {e}")
                continue
            
            # Save checkpoints every 100 entries
            if idx % 100 == 0 or idx == len(messages):
                print(f"💾 Processed {idx}/{len(messages)} emails... (Autosaving Progress)")
                save_to_excel(all_email_data, output_file)

        print(f"\n🎉 Finished processing! File saved cleanly with {len(all_email_data)} rows: {output_file}")

    except HttpError as error:
        print(f"An API error occurred: {error}")
    except Exception as e:
        print(f"An unexpected script error occurred: {e}")

if __name__ == '__main__':
    extract_all_emails()
