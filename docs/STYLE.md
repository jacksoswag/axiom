# Code style

How to write code that reads like I wrote it by hand. This file covers the places where AI
defaults and my habits reliably differ; anything it doesn't mention, handle normally. Examples
are Rust; the principles port to any language and any repo.

---

## How to read this file

- **Every rule is a principle. Every example is an invented illustration.** The identifiers,
  numbers, and shapes inside examples show a form; they are never an instruction to use, keep,
  or restore a specific name. If you catch yourself citing this file to defend a particular
  identifier, you have misread it.
- **Current code outranks this file on names and vocabulary.** When the code and an example here
  disagree about what something is called, the code wins. When I ask for a rename, the request
  wins. Never argue against a change because this document appears to bless the old form.
- This file describes form, never content, so it applies to any state of this repo and to other
  repos too.

---

## 1. Never run a formatter

The mistake most likely to destroy work, so it comes first.

Do not run cargo fmt or any equivalent, do not enable format-on-save, and do not "fix
formatting" as a side errand. If a diff shows whitespace-only changes to lines you did not
deliberately edit, you ran a formatter; revert it. My hand-written files routinely exceed a
formatter's default width cap, which is also the quickest way to spot which files in a repo are
hand-written.

Leave stray trailing whitespace, missing end-of-file newlines, and the odd typo in a comment
exactly where they are. They are evidence a human wrote the file. Never open a diff just to fix
one.

What a formatter would do, and the correct form it would destroy:

| a formatter would                      | correct form                                              |
| -------------------------------------- | --------------------------------------------------------- |
| expand a one-line struct               | `pub struct Rng { state: u64 }`                           |
| expand a one-line fn body              | `pub fn len(&self) -> usize { self.items.len() }`         |
| split statements sharing a line        | `x ^= x >> 12; x ^= x << 25; x ^= x >> 27;`               |
| expand an inline if/else value         | `if limit == 0 { 0 } else { (self.next() % limit) as usize }` |
| collapse an aligned comment column     | see section 3                                             |
| wrap a coherent long chain             | keep it on one line to ~118 characters                    |
| strip trailing spaces, add EOF newline | leave both alone                                          |

---

## 2. Density and layout

- **Write dense.** My hand-written Rust averages 40 to 55 characters per line; formatter-shaped
  output averages about 30. Same complexity, fewer lines.
- **Put a whole small thing on one line**: a one-field struct, a getter, a short if/else that
  produces a value.
- **Group statements that are one operation onto one line.** Three xorshift rounds are one
  shuffle; two draws feeding one formula are one setup.
- **Let code run to ~118 characters, comments to ~128.** When a chain genuinely will not fit,
  break after the receiver and keep the first call attached, or drop the whole body to a
  continuation line.
- **Vertical whitespace is scarce**: 6 to 10 percent blank lines. None between methods in an
  impl, none after an opening brace, none before a closing one. One blank line between free
  functions.
- **Put a constant next to its user**, directly below the function that reads it, never hoisted
  to the top of the file.

---

## 3. Comments

Comment density tracks how subtle the logic is, never file size. Dense math can carry a comment
on nearly every line; plumbing carries almost none. The kind matters more than the amount.

- **Module header: one to four contiguous header lines.** What the file is, where it helps, and
  what it deliberately is not. No blank separator lines inside it, no essay. A multi-paragraph
  comment above any single item is the anti-pattern.
- **Explain math term by term, in an aligned column inside the expression.** The signature move,
  and the thing a formatter destroys first (invented example):

```rust
let term = // this shell's contribution at this distance
    (-((x - shell.peak).powi(2)     // squared gap between distance and the shell's peak
    / (2.0 * shell.width.powi(2)))) // scale gap penalty by shell's width (wider -> lower penalty)
    .exp() * shell.amp;             // decay from 1->0, then scale by the shell's amplitude
```

- **Explain the intuition, never name the algorithm.** Describe what the code does in domain
  terms; a reader should understand the line without knowing the textbook name for it.
- **A doc comment must earn its place.** Constructors, len, is_empty get nothing; their
  signatures say it all. A doc comment exists to say what the signature cannot: why the item
  exists, or a load-bearing constraint a future editor would break. Never write "Returns the
  length."
- **Struct fields take trailing //**, lowercase, no period, one space before the slashes, never
  column-aligned, and only where the name alone is not enough.
- **Plain ASCII only.** No backticks, no markdown emphasis, no characters that aren't on a
  keyboard. ASCII arrows for consequence (wider -> lower penalty), occasional CAPS for emphasis,
  x and sq for times and squared.
- **Tone**: present tense, compressed, direct. American spellings (center, normalize, neighbor).
- **Comment the reason, never the mechanics**, with one carve-out: match the file's comment
  mode. In code written while I'm still learning the language, line-level mechanical comments
  stay, even where they restate the obvious to a fluent reader; they're a comprehension aid
  until the idiom is internalized. In mature code, cut any comment that restates its line. When
  unsure, match the density and kind of comments already around your edit. A comment that is
  plain wrong or trails off unfinished gets cut in either mode.

---

## 4. Naming

- **Full words for locals in any multi-line body**: weight, counts, order, index, destination.
  Never idx, cnt, tmp, val, res.
- **Single letters only for a pure index in a tight loop or a one-line closure param.** In a
  multi-line closure, spell it out.
- **A count is thing_count.** A bare plural names a collection, so a reader can assume a plural
  iterates and a _count divides.
- **Name a function for what it produces or guarantees.** Never calculate_, get_, compute_. And
  never overpromise: a walker that yields unfiltered candidates is visit_candidates; calling it
  visit_valid claims a filter it doesn't perform.
- **Don't restate the owning type in a method name.** Type::derive, never Type::derive_type.
- **Prefer the plain word over the technical one** when both are accurate for the value in hand.
- **Use the repo's existing domain vocabulary, one word per concept.** Do not invent synonyms
  for a concept the code already names. The vocabulary itself is mine to evolve: the rule is
  consistency with the code as it stands today, never loyalty to any particular word, including
  any word this file uses in an example.
- **Keep the flavor.** Vivid domain words are a feature; do not sand them down into generic
  terms because they look informal.
- **Constructors return the type name or Self, whichever reads better.** Both exist in my code;
  do not normalize one to match the other.

---

## 5. Logic and abstraction

- **Prefer a longer function to an indirection.** A 30-line function doing three phases for one
  caller stays one function; splitting it names three concepts nobody else calls and costs the
  reader two jumps.
- **Extract only when it deletes repetition at real call sites.** A helper called once from one
  spot doesn't clear the bar; inline it. The same logic duplicated across two files does clear
  it, pulled up to wherever both can reach. Place a helper immediately next to its caller, no
  blank-line ceremony.
- **No traits, no generics, no lifetimes, no builders** in application code. Concrete types and
  free functions. A trait with one implementor is a finding against you.
- **No Result or Option for programmer error.** Let an out-of-range index panic. Reserve Result
  for real boundaries: parsing files, bytes off disk, the network.
- **Do not assert.** Validation belongs at a trust boundary, once, never at three layers.
- **Handle an edge case where it arises, in one inline clause**: .min(1.0) on a draw, .max(1e-7)
  before a log, % n to wrap a seam. One operator, no branch, no helper, no comment unless the
  reason is invisible.
- **Guards are expressions when short**, never early returns.
- **Derive the minimum.** No reflexive Debug, PartialEq, or Default without a caller.
- **State performance reasoning in prose where the reader hits it.** No #[inline], no unsafe, no
  unexplained micro-optimization.
- **No `#[cfg(test)]` module anywhere under src/, ever.** Tests live in `tests/` outside src/,
  exercising public behavior, and earn their place by failing for a reason that matters.

---

## 6. AI habits to suppress

Each of these is a default the models keep reaching for, absent from my hand-written code, and a
defect if added:

- multi-paragraph doc essays (a constant gets one line)
- error enums, Display impls, or variants nothing constructs
- validation repeated at more than one layer
- Option around a thing that cannot be absent
- pub(crate) ceremony (items are pub or private)
- a builder for a two-field struct
- reflexive derives or a Default impl nothing calls
- a `#[cfg(test)]` module in any src/ file (tests live in `tests/`)
- blank-line padding between methods
- British spellings
- markdown, backticks, or non-keyboard characters inside comments
- "not X but Y" phrasing in comments
- comments that restate the next line
- renaming toward blandness: sanding a vivid domain word into a generic one
- treating documentation as spec: docs describe, code decides; when they disagree the code is
  right and the doc is stale
- "fixing" deliberate style: typos in comments, trailing whitespace, an unusual but working form

---

## 7. Before you finish

1. Ran a formatter, even by accident? Revert it.
2. Put your file next to the densest hand-written file in the repo. Visibly airier at the same
   complexity? Tighten.
3. Every doc comment: does it say something the signature does not? Delete the rest.
4. Every guard, assert, Option, Result: at a real trust boundary? Delete the rest.
5. Every new type, trait, or helper: does it delete more code than it adds? Inline the rest.
6. About to cite this file to justify keeping an old name? Reread the top.
