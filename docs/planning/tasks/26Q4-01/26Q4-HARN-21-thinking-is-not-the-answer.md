### 26Q4-HARN-21: A thinking model's scratch work is not its answer

**User Story**: As the person judging whether committee mode works on the served model, I want the harness to read only a reply's answer, so that a working group's blackboard holds what a member proposed and a session's prompts don't grow by every member's thinking.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-21-thinking-is-not-the-answer` |
| **Dependencies** | None. Stacked on `26Q4-CHORE-02` so one commit carries both for the box. Before R6, R7 and every committee arm |
| **Gate** | full |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `4b01509`):
- `Qwen/Qwen3.6-27B-FP8` opens every reply inside a thinking block. The chat template supplies the opening tag, so the server returns scratch work, `</think>`, then the answer, all in `message.content`.
- Round R5e's run (job 352, flat iterative mode): six replies, each with exactly one `</think>`. The first reply was 30,674 characters, 26,703 of them thinking. Thinking held 50 of the run's 54 Lean code fences. Every `<artifact>` block was in an answer, so flat mode's extraction happened to read the right text.
- Committee mode has no such luck. `WorkingGroup._update_blackboard` takes the first Lean fence in a reply, `_parse_artifact` the first `<artifact>`, `_has_consensus` any mention of the phrase, and `_build_context` pastes every earlier reply into the next prompt.
- `lms/config.py` caps a completion at 64,000 tokens and notes that vLLM refuses a request whose prompt plus cap exceeds the 131,072-token window. A session that carries about 10,000 tokens of thinking per reply reaches that in a few turns.
- The August committee runs used a model that does not think, so none of this showed.

**Investigation**:
```bash
grep -rn "</think>\|reasoning_content" lms/        # nothing
grep -n "re.search" lms/working_group.py           # first match, whole reply
```

---

#### Acceptance Criteria

- [ ] `lms/providers/base.py::split_thinking` returns (thinking, answer), splitting after the last `</think>`; a reply with no closing tag is all answer
- [ ] `GenerationResponse.thinking` (default `""`) carries the scratch work; `content` is the answer only
- [ ] `OpenAIProvider.generate` splits every reply, and keeps `reasoning_content` as thinking when a server has already separated it
- [ ] Token usage is unchanged: thinking tokens are still counted
- [ ] Through a working group with the real provider: the blackboard holds the proposal, no later prompt carries an earlier reply's thinking, the artifact is the scribe's answer, and consensus is not read out of thinking: `uv run pytest tests/test_thinking.py -q`
- [ ] A committee run on the box gets past generation 0 without a context-length refusal, and its blackboards hold proposals — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/providers/base.py` | MODIFY | `split_thinking`, `GenerationResponse.thinking` |
| `lms/providers/openai.py` | MODIFY | Split every reply |
| `tests/test_thinking.py` | CREATE | The split, the provider, a working group end to end |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-21-thinking-is-not-the-answer.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-21.sh` | CREATE | Verify script |

---

#### Notes

- **The thinking is not saved.** `conversations.json` records `content`, so transcripts now hold answers only. Keeping the thinking beside them needs the trace store's call sites to pass it; that is a follow-up, and `GenerationResponse.thinking` is where it would come from.
- **A reply cut off while thinking** has no closing tag and still reads as an answer. It cannot be told from a model that does not think without knowing the model.
- **R5e's own number changes meaning.** Its 57,843 tokens include the thinking. Nothing about its cost is wrong, but any earlier run of this model in committee mode would have measured the fault above.
- Serving with a vLLM reasoning parser would also separate the thinking. It was not used here because the parser's name for this model is untested on the box, and a wrong one stops the server.
