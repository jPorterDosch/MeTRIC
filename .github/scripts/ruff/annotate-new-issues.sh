#!/usr/bin/env bash
# =============================================================================
# Annotate NEW Issues Only
# =============================================================================
# Generates GitHub annotations for the exact occurrences collect-stats.sh
# identified as new (new_issues_precise.json), not just "any occurrence of a
# rule/file that regressed somewhere" — rule-level and file-level filtering
# both over-annotate: a rule-level delta only proves SOME occurrence of that
# rule is new (not which one), and a file-level filter misattributes an
# entire touched file — including pre-existing issues merely shifted by
# unrelated edits, or moved wholesale from another path — as new.
#
# Input files (from collect-stats.sh):
#   - new_issues_precise.json: exact new-issue occurrences (JSON array)
#
# Environment:
#   - GITHUB_WORKSPACE: checkout directory (for stripping absolute paths)
#
# Output:
#   - GitHub annotations (::error file=...,line=...) printed to stdout
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

D=".ruff-stats"

log_section "Annotating new issues"

# Exit silently if no new issues
if [ ! -s "$D/new_issues_precise.json" ] || [ "$(jq 'length' "$D/new_issues_precise.json")" -eq 0 ]; then
  echo "  ✓ No new issues to annotate"
  exit 0
fi

PRECISE_COUNT=$(jq 'length' "$D/new_issues_precise.json")
echo "  → Precise new-issue occurrences: $PRECISE_COUNT"

# Check environment
if [ -z "${GITHUB_WORKSPACE:-}" ]; then
  echo "  ⚠️  GITHUB_WORKSPACE not set (paths may be absolute)"
fi

# Get workspace prefix for path stripping (with trailing slash)
WS_PREFIX=$(get_workspace_prefix)

# Generate annotations and count them
# Store in temp file to both output and count
TEMP_ANNOTATIONS="$D/annotations.txt"
jq -r --arg ws "$WS_PREFIX" '
  .[] |
  "::error file=\(.filename | ltrimstr($ws)),line=\(.location.row),col=\(.location.column)::\(.code): \(.message)"
' "$D/new_issues_precise.json" > "$TEMP_ANNOTATIONS"

ANNOTATION_COUNT=$(wc -l < "$TEMP_ANNOTATIONS" | tr -d ' ')
echo "  → Annotations: $ANNOTATION_COUNT"

# Output annotations to stdout (for GitHub to pick up)
cat "$TEMP_ANNOTATIONS"
