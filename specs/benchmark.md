# Specification: Stacks-Bench (LMS-Bench)

**Status**: Draft
**Date**: 2025-12-18
**Goal**: Establish the first rigorous benchmark for "Long-Horizon Multi-Agent Formalization" of large mathematical libraries.

---

## 1. Motivation: Beyond "One-Shot" Proving

Existing benchmarks (MiniF2F, ProofNet, PutnamBench) measure an AI's ability to solve a single, isolated problem. They fail to capture the complexity of **building a library**:
*   **Consistency**: Defining `Category` in Gen 1 and reusing it in Gen 50.
*   **Dependency Management**: Handling deep chains of prerequisites (Tag A → Tag B → Tag C).
*   **Maintenance**: Preventing "Definition Drift" (redefining existing concepts).

**Stacks-Bench** simulates the real-world task of a mathematician: "Here is Chapter 1. Please formalize Chapter 2 using only the definitions from Chapter 1."

## 2. Dataset & Task Definition

The benchmark is derived from the **Stacks Project** (algebraic geometry), specifically its GitHub repository.

### 2.1 The Dependency Graph
We utilize the raw LaTeX files and the `tags/` directory from the Stacks Project repository.
*   **Source**: [stacks/stacks-project](https://github.com/stacks/stacks-project)
*   **Data**: The `tags` file maps 4-character codes to LaTeX labels. The LaTeX files contain `\ref{}` calls which define the graph edges.

### 2.2 The Task
**Input**:
*   `Foundation.lean`: A preamble containing all dependencies *prior* to the target chapter.
*   `Target`: A set of tags (e.g., "All definitions and lemmas in Section 10.1").

**Constraint**:
*   The agent(s) must produce valid Lean 4 code.
*   The code must *only* import `Foundation.lean` (and standard Mathlib).
*   The code must be accepted by the Lean compiler.

## 3. Metrics

We evaluate performance across three dimensions: **Progress**, **Efficiency**, and **Cohesion**.

### 3.1 Hero Metrics (Progress)
*   **Coverage Score ($C$)**: The deepest "depth" in the dependency graph successfully verified.
    *   *Why*: Harder theorems are deeper in the graph.
*   **Pass Rate ($P$)**: `(Verified Artifacts) / (Total Proposed Artifacts)`.
    *   *Why*: Measures reliability. A system that guesses 100 times to get 1 right is poor.

### 3.2 Efficiency Metrics (Cost)
*   **Compute Efficiency ($E$)**: `(Total Output Tokens) / (Verified Lean Lines)`.
    *   *Why*: Measures the "verbosity cost" of the intelligence.
*   **Correction Overhead**: Average number of verification round-trips required per success.

### 3.3 Social/Quality Metrics (Cohesion)
*   **Citation Density**: `(References to Existing Artifacts) / (Total Artifacts)`.
    *   *Why*: Measures "standing on the shoulders of giants." Low density = reinventing the wheel.
*   **Definition Drift (The "Bloat" Metric)**:
    *   Formula: $1 - \frac{\text{Unique Concepts Formalized}}{\text{Total Structures Defined}}$
    *   *Why*: If 5 agents define 5 versions of `Ring`, drift is high. We want 1 definition used 5 times.

## 4. Benchmark Modes

### Mode A: "The Lone Genius" (Baseline)
*   **Setup**: Single Agent (e.g., Claude Opus, GPT-4, or Llama-3).
*   **Memory**: Infinite context (or RAG).
*   **Goal**: How far can one context window go before collapsing?

### Mode B: "The Society" (Multi-Agent)
*   **Setup**: $N$ Agents, $M$ Generations.
*   **Communication**: Only via `Foundation.lean` (and optional Chat).
*   **Goal**: Does adding agents increase Coverage (better search) or decrease Efficiency (coordination overhead)?

## 5. Required Reading & References

To structure this benchmark effectively, the following papers/resources are essential:

### On Auto-Formalization
1.  **"ProofNet: Autoformalizing and Formally Proving Undergraduate Mathematics" (Azerbayev et al., 2023)**
    *   *Relevance*: Standard baseline for mapping natural language to Lean.
2.  **"MiniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics" (Zheng et al., 2022)**
    *   *Relevance*: Understanding the "one-shot" evaluation paradigm (what we are trying to move beyond).
3.  **"Llemma: An Open Language Model for Mathematics" (Azerbayev et al., 2023)**
    *   *Relevance*: The state-of-the-art open model for math. Essential if we use local models.

### On Multi-Agent Software Engineering
4.  **"Swe-bench: Can Language Models Resolve Real-World GitHub Issues?" (Jimenez et al., 2024)**
    *   *Relevance*: The closest analogy to what we are doing. They measure agents editing a complex codebase. We measure agents extending a complex math library.
5.  **"Communicative Agents for Software Development" (ChatDev) (Qian et al., 2023)**
    *   *Relevance*: seminal paper on role-playing agents (CEO, CTO, Coder) building software.

### On The Stacks Project Structure
6.  **"The Stacks Project" (de Jong et al.)**
    *   *Relevance*: Using the raw LaTeX source and `tags` system as the ground truth.

## 6. Implementation Plan

1.  **Ingestion**: User clones `stacks-project` repo. We write `scripts/parse_stacks_tags.py` to extract the dependency graph from LaTeX sources.
2.  **Harness**: Build `benchmark/harness.py` to run `Society` against the parsed dataset.
3.  **Visualizer**: A dashboard showing the "Frontier of Formalization" moving through the graph.
