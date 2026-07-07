"""
download.py
===========
One-time setup script (run manually, not part of the app's runtime path).
Downloads the base LegalBERT model/tokenizer from Hugging Face and saves a
local copy to disk, so legal_core.load_models() can load it from a local
path instead of re-downloading from the Hub on every app startup.

NOTE (portability / deployability): `save_path` is currently a hardcoded
path specific to one developer's machine. Both this script and
legal_core.py's `load_models()` must agree on this path for the app to run.
For a shareable/deployable setup, this should be replaced with a relative
path (e.g. "./models/legalbert") or an environment variable
(e.g. os.environ.get("LEGALBERT_PATH", "./models/legalbert")) read by both
files, so a fresh clone of the repo works out of the box after running this
script once.
"""

from transformers import AutoTokenizer, AutoModel

# Base (not fine-tuned) LegalBERT checkpoint from the Hugging Face Hub.
model_name = "nlpaueb/legal-bert-base-uncased"

# Hardcoded local path — see portability note above. Must match the
# `model_path` used in legal_core.load_models().
save_path = r"/Users/balamuralibr/legalbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Persist both tokenizer and model config/weights to `save_path` so later
# runs load from disk instead of hitting the Hugging Face Hub every time.
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)