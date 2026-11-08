#!/usr/bin/env bash
# Sole-committer audit.
#
# Fails if any commit reachable from any ref:
#   * has an author or committer other than the canonical identity,
#   * carries a git trailer (Co-authored-by, Signed-off-by, ...),
#   * contains AI/tool attribution filler in its message,
#   * has a subject that is not a Conventional Commit.
# Also fails if the repo-local identity would produce a non-canonical commit.
set -euo pipefail

readonly EXPECTED_NAME="mikeZ98"
readonly EXPECTED_EMAIL="145975972+mikeZ98@users.noreply.github.com"
readonly CONVENTIONAL_RE='^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([a-z0-9._-]+\))?!?: [^ ].*[^. ]$'
readonly FILLER_RE='co-authored-by|signed-off-by|generated with|noreply@anthropic|claude|chatgpt|copilot|openai|ai[- ]generated'

cd "$(git rev-parse --show-toplevel)"

failures=0
fail() {
    printf 'FAIL: %s\n' "$1" >&2
    failures=$((failures + 1))
}

# 1. Identity that the *next* commit will use.
configured_name="$(git config user.name || true)"
configured_email="$(git config user.email || true)"
[ "$configured_name" = "$EXPECTED_NAME" ] \
    || fail "configured user.name is '${configured_name}', expected '${EXPECTED_NAME}'"
[ "$configured_email" = "$EXPECTED_EMAIL" ] \
    || fail "configured user.email is '${configured_email}', expected '${EXPECTED_EMAIL}'"

# 2. History.
if ! git rev-parse --verify --quiet HEAD >/dev/null; then
    printf 'attribution: no commits yet; identity configuration checked only\n'
else
    commits="$(git rev-list --all)"
    count=0
    for sha in $commits; do
        count=$((count + 1))
        short="$(git rev-parse --short "$sha")"

        an="$(git show -s --format='%an' "$sha")"
        ae="$(git show -s --format='%ae' "$sha")"
        cn="$(git show -s --format='%cn' "$sha")"
        ce="$(git show -s --format='%ce' "$sha")"
        [ "$an" = "$EXPECTED_NAME" ] || fail "${short}: author name '${an}'"
        [ "$ae" = "$EXPECTED_EMAIL" ] || fail "${short}: author email '${ae}'"
        [ "$cn" = "$EXPECTED_NAME" ] || fail "${short}: committer name '${cn}'"
        [ "$ce" = "$EXPECTED_EMAIL" ] || fail "${short}: committer email '${ce}'"

        trailers="$(git show -s --format='%(trailers:only,unfold)' "$sha")"
        [ -z "$trailers" ] || fail "${short}: commit carries trailers: ${trailers}"

        message="$(git show -s --format='%B' "$sha")"
        if printf '%s' "$message" | grep -Eiq "$FILLER_RE"; then
            fail "${short}: commit message contains attribution filler"
        fi

        subject="$(git show -s --format='%s' "$sha")"
        if ! printf '%s' "$subject" | grep -Eq "$CONVENTIONAL_RE"; then
            fail "${short}: subject is not a Conventional Commit: '${subject}'"
        fi
    done
    printf 'attribution: audited %d commit(s)\n' "$count"
fi

if [ "$failures" -ne 0 ]; then
    printf 'attribution: %d violation(s)\n' "$failures" >&2
    exit 1
fi
printf 'attribution: OK\n'
