"""
CRUD operations package.
"""

# Import from base crud module
from ..base_crud import (
    get_assistant, get_assistants, create_assistant, update_assistant, delete_assistant,
    create_call_log, get_call_log, get_call_logs, update_call_log, delete_call_log,
    create_phone_number, get_phone_number, get_phone_numbers, update_phone_number, delete_phone_number
)

# Import from campaign crud module
from .campaign_crud import (
    create_campaign, get_campaign, get_campaigns, get_campaigns_count, update_campaign, delete_campaign,
    start_campaign, pause_campaign, resume_campaign, stop_campaign,
    create_campaign_contact, get_campaign_contacts, update_campaign_contact
)
