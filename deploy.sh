#!/usr/bin/env bash
# Deploy to Hugging Face Spaces (Docker SDK).
#
#   HF_TOKEN=hf_xxx HF_USER=yourname ./deploy.sh
#
# The token is read from the environment and used only for this push. It is
# never written to a file, and `git remote` is configured without it so it
# cannot end up in .git/config.
set -euo pipefail

SPACE_NAME="${SPACE_NAME:-gulf-property-ai}"
: "${HF_TOKEN:?set HF_TOKEN (https://huggingface.co/settings/tokens, type=Write)}"
: "${HF_USER:?set HF_USER (your Hugging Face username)}"

SPACE_ID="${HF_USER}/${SPACE_NAME}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"
LIVE_URL="https://${HF_USER//./-}-${SPACE_NAME}.hf.space"

echo "==> Target Space : ${SPACE_ID}"
echo "==> Live URL     : ${LIVE_URL}"

# --- 1. create the Space if it does not exist -----------------------------
echo "==> Creating Space (ignored if it already exists)"
python - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
sid = f'{os.environ["HF_USER"]}/{os.environ.get("SPACE_NAME", "gulf-property-ai")}'
try:
    api.create_repo(repo_id=sid, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"    Space ready: {sid}")
except Exception as e:
    print(f"    create_repo: {e}", file=sys.stderr)
    sys.exit(1)
PY

# --- 2. set the API key as a SECRET (not a variable) ----------------------
# Variables are visible in build logs and to the client; secrets are not.
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  echo "==> Setting OPENROUTER_API_KEY as a Space secret"
  python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
sid = f'{os.environ["HF_USER"]}/{os.environ.get("SPACE_NAME", "gulf-property-ai")}'
api.add_space_secret(repo_id=sid, key="OPENROUTER_API_KEY",
                     value=os.environ["OPENROUTER_API_KEY"])
api.add_space_variable(repo_id=sid, key="PUBLIC_URL",
                       value=os.environ.get("LIVE_URL", ""))
print("    secret set")
PY
else
  echo "==> WARNING: OPENROUTER_API_KEY not set - Space will run in search mode"
fi

# --- 3. push ---------------------------------------------------------------
echo "==> Pushing to the Space"
git remote remove space 2>/dev/null || true
git remote add space "${SPACE_URL}.git"

# Credentials are passed for this invocation only, never persisted.
git -c credential.helper= \
    -c "http.${SPACE_URL}.extraheader=Authorization: Basic $(printf "user:%s" "${HF_TOKEN}" | base64 -w0)" \
    push --force space HEAD:main

git remote remove space

echo
echo "=========================================================="
echo " Pushed. The Space is now building (first build ~15 min)."
echo
echo "   Build logs : ${SPACE_URL}"
echo "   Live URL   : ${LIVE_URL}"
echo
echo " Check readiness with:"
echo "   curl -s ${LIVE_URL}/healthz"
echo "=========================================================="
