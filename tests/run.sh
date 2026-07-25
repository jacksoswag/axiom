#!/usr/bin/env bash
# The three suites, one flag each. Every run writes a dated report into tests/reports and prints it
# back in a form worth reading. Nothing here knows what any case does: the suites own that, this owns
# running them, naming what they wrote, and rendering it.
#
#   tests/run.sh --smk           the fast in-process suite
#   tests/run.sh --behave        the harness end to end, over its own protocol and the relay
#   tests/run.sh --perf          what the machine costs, minutes of real simulation, release build
#   tests/run.sh --smk --behave  any combination, in the order written here

set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root" || exit 1
command -v cargo >/dev/null 2>&1 || PATH="/opt/homebrew/bin:$PATH"
command -v cargo >/dev/null 2>&1 || { echo "no cargo on PATH"; exit 127; }

stamp=$(date +%Y-%m-%d)
reports="$root/tests/reports"
logs="$reports/.logs"
mkdir -p "$logs"

if [ -t 1 ]; then
    bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; green=$'\033[32m'; reset=$'\033[0m'
    width=$(tput cols 2>/dev/null || echo 100)
else
    bold=; dim=; red=; green=; reset=; width=100
fi
[ "$width" -lt 60 ] && width=60

usage() {
    cat <<'ASKED'
usage: tests/run.sh [--smk] [--behave] [--perf]

  --smk     the fast suite: engine, tuner and harness behavior in process
  --behave  the harness end to end: a real process over its own protocol, plus the relay
  --perf    timings and determinism, release build, minutes of real simulation

Each flag writes tests/reports/DATE_{smk,behave,perf}.md and prints it.
ASKED
}

# One markdown report, printed for a person: headings stand out, tables line up inside the terminal,
# and a broken promise is the one thing coloured.
render() {
    local file=$1
    [ -f "$file" ] || { printf '%s\n' "  ${red}no report was written to $file${reset}"; return 1; }
    awk -v bold="$bold" -v dim="$dim" -v red="$red" -v green="$green" -v reset="$reset" -v cap="$width" '
        function blanks(count,   out) { out = ""; while (count-- > 0) out = out " "; return out }
        function fit(   column, widest, over, room) {
            for (column = 1; column <= columns; column++) wide[column] = 0
            for (row = 1; row <= rows; row++)
                for (column = 1; column <= columns; column++)
                    if (length(cell[row, column]) > wide[column]) wide[column] = length(cell[row, column])
            while (1) {
                over = 2 - cap
                for (column = 1; column <= columns; column++) over += wide[column] + 2
                if (over <= 0) return
                widest = 1
                for (column = 2; column <= columns; column++) if (wide[column] > wide[widest]) widest = column
                if (wide[widest] <= 24) return              # every column is already as narrow as it reads
                room = wide[widest] - over
                wide[widest] = (room > 24) ? room : 24
            }
        }
        function table(   row, column, line, text, plain, rule) {
            if (rows == 0) return
            fit()
            for (row = 1; row <= rows; row++) {
                line = "  "
                for (column = 1; column <= columns; column++) {
                    text = cell[row, column]
                    if (length(text) > wide[column]) text = substr(text, 1, wide[column] - 3) "..."
                    plain = text
                    if (text == "yes") text = green text reset
                    else if (text == "NO") text = red text reset
                    else if (row == 1) text = bold text reset
                    line = line text blanks(wide[column] - length(plain) + 2)
                }
                sub(/[ ]+$/, "", line)
                print line
                if (row == 1) {                            # a rule under the header, as wide as the table
                    rule = ""
                    for (column = 1; column <= columns; column++) rule = rule blanks(wide[column] + 2)
                    gsub(/ /, "-", rule)
                    print "  " dim rule reset
                }
            }
            rows = 0; columns = 0
            print ""
        }
        /^\|/ {                                            # a table row, buffered until the table ends
            if ($0 ~ /^\|[ -:|]+\|$/) next                 # markdown alignment row, redrawn as a rule
            line = $0
            gsub(/\\\|/, "\001", line)                     # an escaped pipe is content, not a column edge
            sub(/^\|/, "", line); sub(/\|[ ]*$/, "", line)
            count = split(line, parts, "|")
            rows++
            if (count > columns) columns = count
            for (column = 1; column <= count; column++) {
                text = parts[column]
                gsub(/^[ ]+|[ ]+$/, "", text)
                gsub(/\001/, "|", text)
                cell[rows, column] = text
            }
            next
        }
        { table() }                                        # anything else ends the table it followed
        /^# / { sub(/^# /, ""); print "\n" bold toupper($0) reset; next }
        /^## / { sub(/^## /, ""); print bold $0 reset; next }
        /^$/ { print ""; next }
        { print "  " $0 }
        END { table() }
    ' "$file"
}

# The fast suite reports through cargo rather than through a file of its own, so the report is built
# here out of what the test runner said: one row per case, and the panic message for anything broken.
smoke_report() {
    awk -v stamp="$1" '
        /^test [A-Za-z0-9_:]+ \.\.\. / {
            name = $2; state = $NF
            order[++seen] = name; held[name] = state
            if (state != "ok") broken++
            next
        }
        /^---- .* stdout ----$/ { current = $2; next }
        /panicked at/ { if (current != "") grab = 1; next }
        grab && NF { note[current] = $0; grab = 0; next }
        END {
            printf "# Smoke, %s\n\n", stamp
            printf "%d cases checked, %d broken. Each one is a single promise, checked in process\n", seen, broken
            printf "against the library rather than against a running harness.\n\n"
            for (at = 2; at <= seen; at++) {            # cargo runs them side by side, so the report sorts them
                held_name = order[at]
                for (back = at - 1; back >= 1 && order[back] > held_name; back--) order[back + 1] = order[back]
                order[back + 1] = held_name
            }
            for (at = 1; at <= seen; at++) {
                name = order[at]
                split(name, path, "::")
                if (path[1] != area) {
                    if (area != "") printf "\n"
                    area = path[1]
                    printf "## %s\n\n| held | case |\n|---|---|\n", area
                }
                case_name = path[2]; gsub(/_/, " ", case_name)
                printf "| %s | %s |\n", (held[name] == "ok" ? "yes" : "NO"), case_name
            }
            if (broken > 0) {
                printf "\n## broken\n\n| case | what it said |\n|---|---|\n"
                for (at = 1; at <= seen; at++) {
                    name = order[at]
                    if (held[name] == "ok") continue
                    said = note[name]; gsub(/\|/, "\\|", said)
                    printf "| %s | %s |\n", name, said
                }
            }
        }
    '
}

failed=0
announce() { printf '\n%s\n' "${bold}==> $1${reset}"; }

smk() {
    announce "smoke: the fast suite"
    local log="$logs/smk.log" out="$reports/${stamp}_smk.md"
    cargo test --test smoke > "$log" 2>&1
    local code=$?
    smoke_report "$stamp" < "$log" > "$out"
    if ! grep -q '^test result:' "$log"; then                # it never got as far as running anything
        printf '%s\n' "  ${red}the suite did not build${reset}"
        tail -30 "$log" | sed 's/^/  /'
        failed=1
        return
    fi
    render "$out"
    printf '  %s\n' "${dim}$out${reset}"
    [ $code -ne 0 ] && failed=1
    return 0
}

behave() {
    announce "behavior: the harness end to end"
    local log="$logs/behave.log" out="$reports/${stamp}_behave.md"
    cargo test --test behavior -- --nocapture > "$log" 2>&1
    local code=$?
    if [ ! -f "$out" ]; then
        printf '%s\n' "  ${red}no report was written${reset}"
        tail -30 "$log" | sed 's/^/  /'
        failed=1
        return
    fi
    render "$out"
    printf '  %s\n' "${dim}$out${reset}"
    [ $code -ne 0 ] && failed=1
    return 0
}

perf() {
    announce "performance: minutes of real simulation, release build"
    local log="$logs/perf.log" out="$reports/${stamp}_perf.md"
    cargo test --release --test performance -- --ignored --nocapture > "$log" 2>&1
    local code=$?
    if [ ! -f "$out" ]; then
        printf '%s\n' "  ${red}no report was written${reset}"
        tail -30 "$log" | sed 's/^/  /'
        failed=1
        return
    fi
    render "$out"
    printf '  %s\n' "${dim}$out${reset}"
    [ $code -ne 0 ] && failed=1
    return 0
}

[ $# -eq 0 ] && { usage; exit 2; }
wanted=()
for flag in "$@"; do
    case "$flag" in
        --smk) wanted+=(smk) ;;
        --behave) wanted+=(behave) ;;
        --perf) wanted+=(perf) ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown flag %s\n\n' "$flag"; usage; exit 2 ;;
    esac
done
for suite in "${wanted[@]}"; do "$suite"; done

if [ $failed -eq 0 ]; then printf '\n%s\n' "${green}every promise held${reset}"
else printf '\n%s\n' "${red}something is broken, see the report above${reset}"; fi
exit $failed
