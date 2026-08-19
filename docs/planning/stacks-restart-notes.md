# Stacks restart notes — 2026-08-19

One page on what the Phase 1 kernel plan assumed, what has changed since it was
written (2026-02-14), and what the first committee run against Track B would
actually tell us.

## What Phase 1 assumed (2026-02-14)

The three-text synthesis planned a shared kernel (~350–500 stmts) built first,
in three parallel Phase 1 tracks — A spectral sequences, B fibred categories +
descent, C core algebra — executed by working-committee cycles, building ON
Mathlib, with the Stacks Project as reference formalization. Throughput and
cost projections attached to that plan came from a mock-verified run and are
**retracted**; the plan's *ordering* (kernel before any text-specific work) was
never invalidated and still stands.

## What has changed since

1. **The verification story collapsed and was rebuilt.** Nothing before
   2026-08-10 was real-verified; the first `verified_lean` artifact exists now,
   the oracle runs under `lake env`, provenance is recorded, and accumulated
   work reaches the next generation (HARN-01/-02/-07/-09/-10/-11). The
   remaining gates — T2/T4, novelty, cost — are in flight (HARN-03/-04/-05).
2. **The program is gated on calibration.** Everything through 2026-09-30
   serves one number (CVFN). Stacks formalization *spend* stays deferred until
   it exists; this restart is preparation only (goal specs, no proving).
3. **Committee mode has never actually run.** `PlanningPanel`/`WorkingGroup`
   pass tests but no CLI path reaches them; 26Q3-HARN-12 (in flight) wires
   them and adds the review stage. Every run so far was flat or iterative mode
   against the Ch. 4 warm-up goal — so "2–3 WC cycles per track" has never
   been grounded in an observed cycle.
4. **WC-3 is compiled and real** (0 errors, 1 known-unfillable sorry), so
   Track B genuinely starts from an existing corpus rather than from nothing —
   the one Phase 1 assumption that came out *stronger* than planned.

## What a first committee run on Track B would tell us

- **Allocation**: does the planning panel distribute distinct Track B tags
  across committees, or do N agents converge on one tag as in
  `shakedown_3x3_d`? This is the population-size hypothesis' first
  precondition.
- **Reuse across a real boundary**: Track B statements genuinely need WC-3's
  2-categorical corpus — the first setting where citing the foundation is
  load-bearing rather than optional.
- **Where Mathlib's partial `FiberedCategory.*` coverage actually ends** — the
  N0/N1 classifier (HARN-04) over Track B gives the kernel's novelty profile,
  the same measurement the ANT arcs get for calibration.
- **A grounded WC-cycle observation** to replace the retracted throughput
  figures — one measured cycle, not a projection.

## Not restarted

Phase 2+ (sites, group cohomology, p-adic bridge), Track C, all ART/CNF work,
OCR of `cnf_2nd.pdf`, and any Lean proving against these goals. All still
gated on the calibration verdict.
