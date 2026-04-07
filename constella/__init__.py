"""Constella — multi-agent voice constellation for bilingual healthcare follow-up calls."""

from dotenv import load_dotenv

# Load .env from the project root the moment anything in `constella` is imported.
# Safe to call multiple times. Real OS env vars take precedence over .env values.
load_dotenv(override=False)

__version__ = "0.1.0"
