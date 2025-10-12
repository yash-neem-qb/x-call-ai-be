# Unified Pipeline Test Seeding

This document captures the workflow for populating the database with a rich set of synthetic data that exercises every major subsystem (organizations, users, teams, assistants, telephony objects, calls, and knowledge documents). The new `seed_database.py` utility provides an idempotent, configurable way to install that dataset.

## Pre-requisites
- Python 3.11+
- The project's dependencies installed (`pip install -r requirements.txt`)
- Access to the target PostgreSQL database (connection driven by `.env` and `app/config/settings.py`)

## Running the Seeder
1. Retrieve an authentication payload for the user that should own the seeded data. You can reuse the JSON object returned by the `/auth/login` endpoint. The payload **must** include these keys:
   - `access_token`
   - `token_type`
   - `expires_in`
   - `user_id`
   - `email`
   - `first_name`
   - `last_name`
   - `full_name`
   - `organization_id`
   - `organization_name`

2. Execute the seeder. You can either embed the payload inline or pass a file path:

```bash
python seed_database.py \ 
  --payload-json '{
    "access_token": "<token>",
    "token_type": "bearer",
    "expires_in": 1800,
    "user_id": "88af97c2-fca8-4f03-8b5c-0ef2d5eb79f1",
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "full_name": "John Doe",
    "organization_id": "dfb82ba4-67e1-42e8-9217-dc1f0c05ca4c",
    "organization_name": "My Company"
  }'
```

Use `--payload-file <path>` if you prefer to keep the JSON on disk.

3. Adjust volumes with the optional switches:
   - `--teams` (default `2`)
   - `--assistants` (default `2`)
   - `--phone-numbers` (default `3`)
   - `--calls` (default `10`)
   - `--knowledge-docs` (default `5`)

These flags are clamped to a minimum of `1` to preserve coverage.

## What Gets Seeded
- Organization metadata and settings (plan upgraded to `pro`, compliance defaults)
- Primary user and memberships
- Multiple teams with memberships
- Assistants with voice/LLM/transcription configuration
- Phone numbers with webhook configuration stubs
- Call history entries with realistic metrics and analytics payloads
- Knowledge base documents mapped to the primary assistant

The script is idempotent: running it multiple times keeps the dataset up to date without duplicating records.

## Troubleshooting
- Ensure the `.env` has valid Postgres credentials. The script uses `SessionLocal` from `app/db/database.py`.
- Errors are logged to the console via the existing logging configuration (`app/core/logging.py`).
- On failure the script rolls back partial inserts and exits with code `1`.

For additional customization, edit `seed_database.py` and adjust the template generators near the bottom of the module.


