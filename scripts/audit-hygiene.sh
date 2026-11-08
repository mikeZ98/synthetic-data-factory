#!/usr/bin/env bash
# Repository hygiene audit.
#
# Scope is everything git would ship: tracked files plus untracked files that
# are not ignored, and every path ever recorded in history. Fails on:
#   * absolute host paths (macOS/Linux/Windows home dirs, mount points, temp dirs),
#   * macOS filesystem artifacts (.DS_Store, AppleDouble ._*, __MACOSX, Icon\r),
#   * tool state escaping ./.cache/ (.pytest_cache, .mypy_cache, .hypothesis, ...),
#   * ./.cache/ not being gitignored, or any file under it being tracked.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

failures=0
fail() {
    printf 'FAIL: %s\n' "$1" >&2
    failures=$((failures + 1))
}

# Each pattern is split with a bracket class so this script never matches itself.
readonly HOST_PATH_RE='/[U]sers/|/[V]olumes/|/[h]ome/[A-Za-z0-9._-]+|/[p]rivate/(var|tmp)/|/[v]ar/folders/|/[o]pt/homebrew|[C]:\\{1,2}Users\\{1,2}|[C]:/[U]sers/|~/[L]ibrary/'
readonly MAC_ARTIFACT_RE='(^|/)(\.[D]S_Store|\._[^/]*|__[M]ACOSX|\.[A]ppleDouble|\.[S]potlight-V100|\.[T]rashes|Icon'$'\r'')(/|$)'
readonly STRAY_CACHE_RE='(^|/)(\.[p]ytest_cache|\.[m]ypy_cache|\.[r]uff_cache|\.[h]ypothesis|__[p]ycache__|\.[c]overage(\..*)?|[h]tmlcov|[c]overage\.(xml|json)|\.[b]enchmarks|\.[t]ox|\.[n]ox)(/|$)'

shippable="$(git ls-files --cached --others --exclude-standard)"
history_paths="$(git log --all --format= --name-only 2>/dev/null | sort -u || true)"

# 1. Absolute host paths inside shippable file contents.
if [ -n "$shippable" ]; then
    hits="$(printf '%s\n' "$shippable" | tr '\n' '\0' \
        | xargs -0 grep -IEn "$HOST_PATH_RE" -- 2>/dev/null || true)"
    [ -z "$hits" ] || fail "absolute host paths found:
${hits}"
fi

# 2. macOS artifacts and stray tool caches, in the tree git would ship and in history.
for label in "working tree" "history"; do
    if [ "$label" = "history" ]; then paths="$history_paths"; else paths="$shippable"; fi
    [ -n "$paths" ] || continue
    mac="$(printf '%s\n' "$paths" | grep -E "$MAC_ARTIFACT_RE" || true)"
    [ -z "$mac" ] || fail "macOS artifacts in ${label}:
${mac}"
    stray="$(printf '%s\n' "$paths" | grep -E "$STRAY_CACHE_RE" || true)"
    [ -z "$stray" ] || fail "tool caches outside ./.cache/ in ${label}:
${stray}"
done

# 3. Tool state must not materialize outside ./.cache/ on disk (excluding the venv).
on_disk="$(find . \( -path ./.git -o -path ./.cache -o -path ./.venv \) -prune -o -print \
    | sed 's|^\./||' | grep -E "$STRAY_CACHE_RE" || true)"
[ -z "$on_disk" ] || fail "tool state escaped ./.cache/ on disk:
${on_disk}"

# 4. ./.cache/ itself is ignored and untracked.
git check-ignore -q .cache/ || fail "./.cache/ is not gitignored"
tracked_cache="$(git ls-files -- .cache)"
[ -z "$tracked_cache" ] || fail "files under ./.cache/ are tracked:
${tracked_cache}"

if [ "$failures" -ne 0 ]; then
    printf 'hygiene: %d violation(s)\n' "$failures" >&2
    exit 1
fi
printf 'hygiene: OK\n'
