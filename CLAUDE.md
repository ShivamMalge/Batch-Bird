# Batchbird — Mini Columnar Query Engine (Rust)

Planning-docs stage; no code yet. Before writing any code, read `agents.md` — it contains
hard guardrails (scope cuts that require human sign-off to cross) and the phase workflow.

Doc map:
- `prd.md` — goals, non-goals, the single supported query shape, success criteria
- `architecture.md` — modules, data flow, core types
- `systemDesign.md` — the *why* behind each design decision
- `techstack.md` — toolchain (stable + nightly-gated `simd` feature), crates
- `phases.md` — build order with per-phase Status lines; work strictly phase-by-phase
- `mermaid.md` — reference diagrams
