# Agent Instructions

For any AI coding agent (e.g. Claude Code) working in this repository.

## Before Writing Code
- Read `prd.md`, `architecture.md`, `systemDesign.md`, and `phases.md` first. This project's
  scope cuts are deliberate design decisions, not gaps to "helpfully" fill in.
- Work phase-by-phase per `phases.md`. Do not start a later phase's code before the current
  phase's tests pass.

## Resolved Decisions (previously flagged soft spots)
- `Accumulator<T>` generic over `T`: **confirmed 2026-08-18** with explicit human sign-off.
  Build the generic version in Phase 4; the earlier `f64`-only draft is superseded.
- Toolchain: edition 2024 on **stable** for the default build; `std::simd` is still
  nightly-only (as of 2026-08), so SIMD code lives behind the `simd` cargo feature and is
  built/benched with nightly — see `techstack.md`.
- `GroupKey` is `u64` (dict code widened, or raw i64 bit-cast) — the earlier `u32` draft
  could not hold a raw i64 key. See `systemDesign.md` "Group-By".

## Hard Guardrails — do not implement without explicit human sign-off
- No joins (hash join, join ordering).
- No multi-column `GROUP BY` — `GroupKey` stays a single-value newtype.
- No SIMD applied to string/Utf8 filtering.
- No `Box<dyn Accumulator>` / dynamic dispatch — `Aggregate` stays generic/monomorphized
  unless a query genuinely needs heterogeneous accumulators at plan time.
- No expanding the supported SQL surface beyond:
  `SELECT col1, SUM(col2) FROM t WHERE col3 <op> x GROUP BY col1` (op ∈ {=, <, >}).

If a task seems to require crossing one of these lines, stop and flag it rather than
quietly implementing a workaround.

## Testing Expectations
- Every phase that changes execution behavior needs a correctness test comparing output
  against the naive baseline (Phase 3) on the same input.
- SIMD code (Phase 5) must have a scalar equivalent that passes the same tests; SIMD is an
  optimization path, never the only implementation of a given operation.
- Do not mark a phase complete in `phases.md` (its **Status** line) without tests passing.

## Benchmarking Expectations
- Keep the naive baseline (Phase 3) free of operator abstraction — it exists specifically to
  isolate "row-at-a-time" as a variable. Do not refactor it to share code with the batch engine.
- Profile hash-group's two phases (build-index vs. scatter-accumulate) separately, per
  `systemDesign.md`. Do not report a single combined number for group-by timing.
- Report benchmark results honestly, including cases where an "optimization" (e.g. sort-group)
  turns out slower at the tested scale — do not adjust the benchmark to force a cleaner story.

## Style
- Prefer the dumbest correct implementation for Phase 3; save cleverness for Phases 4-6.
- Comment *why* a scope cut was made where it affects code (e.g. why `Filter` compacts
  instead of passing a selection vector) — the reasoning lives in `systemDesign.md` and should
  be referenced, not re-litigated, in code comments.
