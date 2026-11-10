#!/bin/bash
# q_pin_has must not stop a round when the pattern is there. The go scripts run under pipefail, and
# a `git show | grep -q` pipeline fails there whenever grep finds its match early and exits: git is
# then killed by SIGPIPE on its next write. Big files with an early match show it every time.
#   bash test_pin_has.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=fakebox.sh
source "${FB_HARNESS:-$HERE/fakebox.sh}"

big_file() { { echo "def needle_function(): pass"; seq 1 60000 | sed 's/^/filler line number /'; } > "$1/big.py"; }
FB_SEED_HOOK=big_file
cat > "$FB_ROOT/lms_pinhas_go.sh" <<'GO'
#!/bin/bash
set -euo pipefail
source "$HOME/lms_q_lib.sh"
q_init pinhas
Q_PIN="$1"
for _ in $(seq 1 20); do
    q_pin_has big.py 'needle_function' "the needle"
done
echo "20 of 20 found"
q_pin_has big.py 'no_such_thing_anywhere' "a thing no commit has"
GO

echo "== q_pin_has under pipefail, pattern on the first line of a 1.5 MB file"
fb_new pinhas
fb_install "$FB_ROOT/lms_pinhas_go.sh"
fb_run lms_pinhas_go.sh "$(cat "$D/sha_main")"
has "$D/out" "20 of 20 found"
hasnt "$D/out" "lacks the needle"
has "$D/out" "lacks a thing no commit has; nothing is queued"
rc_is 1
fb_done
