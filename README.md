# oauth2_email_automation.py
"""
OAuth2-based Email Automation for Office 365
Ready to use once you get credentials from IT
"""

import msal
import requests
import json
from datetime import datetime
import base64

class Office365EmailClient:
    def __init__(self, tenant_id, client_id, client_secret, mailbox):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.mailbox = mailbox
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.token = None
        
    def get_token(self):
        """Get OAuth2 token"""
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret
        )
        
        result = app.acquire_token_for_client(scopes=self.scope)
        
        if "access_token" in result:
            self.token = result['access_token']
            print("✅ Token acquired successfully")
            return True
        else:
            print(f"❌ Token acquisition failed: {result.get('error')}")
            return False
    
    def read_emails(self, folder="inbox", unread_only=True):
        """Read emails from mailbox"""
        if not self.token:
            if not self.get_token():
                return []
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Build filter
        filter_query = "$filter=isRead eq false" if unread_only else ""
        
        # Get messages
        url = f"https://graph.microsoft.com/v1.0/users/{self.mailbox}/mailFolders/{folder}/messages"
        if filter_query:
            url += f"?{filter_query}&$top=50"
        else:
            url += "?$top=50"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            messages = response.json().get('value', [])
            print(f"✅ Found {len(messages)} {'unread' if unread_only else ''} emails")
            
            # Parse emails
            parsed_emails = []
            for msg in messages:
                parsed_emails.append({
                    'id': msg['id'],
                    'subject': msg.get('subject', ''),
                    'from': msg.get('from', {}).get('emailAddress', {}).get('address', ''),
                    'body': msg.get('body', {}).get('content', ''),
                    'received': msg.get('receivedDateTime', ''),
                    'isRead': msg.get('isRead', False)
                })
            
            return parsed_emails
        else:
            print(f"❌ Failed to read emails: {response.status_code}")
            print(f"Response: {response.text}")
            return []
    
    def send_email(self, to, subject, body):
        """Send email using Graph API"""
        if not self.token:
            if not self.get_token():
                return False
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Create message
        message = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "Text",
                    "content": body
                },
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to
                        }
                    }
                ]
            }
        }
        
        # Send
        url = f"https://graph.microsoft.com/v1.0/users/{self.mailbox}/sendMail"
        response = requests.post(url, headers=headers, json=message)
        
        if response.status_code == 202:
            print(f"✅ Email sent successfully to {to}")
            return True
        else:
            print(f"❌ Failed to send email: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    def mark_as_read(self, message_id):
        """Mark email as read"""
        if not self.token:
            if not self.get_token():
                return False
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Update message
        url = f"https://graph.microsoft.com/v1.0/users/{self.mailbox}/messages/{message_id}"
        data = {"isRead": True}
        
        response = requests.patch(url, headers=headers, json=data)
        
        return response.status_code == 200

# Test script - ready to use when you get credentials
def test_oauth2_connection():
    """Test the OAuth2 connection once you have credentials"""
    
    # These will come from IT
    tenant_id = "YOUR_TENANT_ID"
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    mailbox = "bizops_txnquery@terrapay.com"
    
    print("🔐 Testing OAuth2 Email Connection")
    print("="*50)
    
    # Create client
    client = Office365EmailClient(tenant_id, client_id, client_secret, mailbox)
    
    # Test token acquisition
    if client.get_token():
        print("\n📥 Testing email reading...")
        emails = client.read_emails(unread_only=True)
        
        if emails:
            print(f"\nSample email:")
            email = emails[0]
            print(f"From: {email['from']}")
            print(f"Subject: {email['subject']}")
            print(f"Received: {email['received']}")
        
        # Test sending
        print("\n📤 Testing email sending...")
        success = client.send_email(
            to="sourav.d@terrapay.com",
            subject="OAuth2 Test Email",
            body="This is a test email sent via Microsoft Graph API using OAuth2 authentication."
        )
        
        if success:
            print("\n✅ All tests passed! OAuth2 setup is working.")
    else:
        print("\n❌ Failed to acquire token. Check credentials.")

if __name__ == "__main__":
    print("This script is ready to test OAuth2 connection.")
    print("\nYou need from IT:")
    print("1. Tenant ID")
    print("2. Client ID (from Azure AD app registration)")
    print("3. Client Secret")
    print("\nOnce you have these, update the values in test_oauth2_connection()")
