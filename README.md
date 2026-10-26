# Exception Router

A multi-agent system that triages supply-chain exception tickets (late shipments, failed deliveries, inventory mismatches, quality flags), finds the root cause with real tool calls against mock TMS/WMS/QMS data, applies written escalation policy, files escalations, and drafts the customer reply.

Each ticket goes either through **one agent** (cheap) or a **three-specialist pipeline** (more careful). A deterministic rule picks the route, and the eval measures whether the extra agents are worth their cost.

Built with Python and the Anthropic Messages API (raw tool-use loop, no agent framework).

---

## Problem

Ops teams get a stream of exception tickets in free text: *"SH-2018 arrived 3 days late AND half the cartons were crushed."* Handling one means:
1. working out what kind of problem(s) it is,
2. pulling data from several systems to find the root cause,
3. checking the policy (escalate? to whom? what priority?),
4. filing the escalation and writing a reply that doesn't promise anything untrue.

Steps 1 and 4 need language understanding. Steps 2 and 3 need correct facts and consistent rule-following. The design question is which parts the LLM should own.

## Architecture

```
ticket ──► TRIAGE (1 forced LLM call: types + IDs)
              │
              ▼
        ROUTING RULE (plain Python)
   multi if: >1 exception type | key account | order ≥ ₹5,00,000
              │
     ┌────────┴──────────────────────────────┐
  single                                   multi
     │                                       │
 RESOLVER                       INVESTIGATOR ─► POLICY ─► COMMUNICATOR
 all tools for the types        read tools      lookup_     create_
 + policy + escalation          only            policy      escalation
     │                          root cause      only        only
     │                                          decide      file + draft
     └───────────────┬───────────────────────────┘
                     ▼
     result: root causes, escalate decision, escalation filed?,
             customer message, internal note, cost stats
```

| Component | File | What it does |
|---|---|---|
| Agent loop | `exception_router/agent.py` | The tool-use while-loop (same protocol as `step1_tool_loop.py`) plus structured output, error handling and a forced finish |
| Tools | `exception_router/tools.py` | 8 tools over mock data: `get_order`, `get_shipment`, `get_delivery_attempts`, `get_carrier_status`, `get_inventory`, `get_quality_inspection`, `lookup_policy`, `create_escalation` (writes to `outbox/escalations.jsonl`) |
| Agents | `exception_router/agents.py` | System prompts, per-agent tool subsets, submit schemas |
| Routing | `exception_router/routing.py` | Deterministic single/multi rule |
| Orchestrator | `exception_router/orchestrator.py` | Wires it together and checks decision vs action |
| Data | `data/world.json`, `data/policies.json`, `data/tickets.json` | Mock systems, written policies, 20 labeled tickets |
| Eval | `eval.py` | Accuracy + cost, with `--force single/multi` ablations |
| Tests | `tests/test_offline.py` | 36 offline tests (no API key): tool math, routing, loop protocol with a fake client, label check |

## Key design decisions

**1. The LLM extracts, code decides the route.**
Triage is one LLM call that turns free text into `{exception_types, shipment_ids, ...}`. The single/multi choice is then a plain `if` on customer tier, order value and number of exception types.
*Why:* routing is a business rule. In code it's auditable, unit-tested, free and never drifts. The LLM only does what code can't, which is reading free text.
*Tradeoff:* routing is only as good as triage. A missed exception type means a wrong route. The eval reports `types` accuracy separately so you can see that.

**2. Tools do the arithmetic.**
`get_shipment` returns `days_late`, `get_inventory` returns `variance_pct`, `get_quality_inspection` returns `defect_rate_pct`, all computed against a fixed simulation date (`today` in `world.json`).
*Why:* in step 1 the model had no current date. It said "I can't tell if it's late," then in the same answer called it "almost certainly late" (see *What broke*). Date math and percentages decide escalations, so they shouldn't depend on the model.

**3. Structured output through a "submit" tool.**
Every agent finishes by calling `submit_<something>` with a JSON schema (enums for root causes, teams and priorities). The tool input *is* the result, so there's no parsing of prose.
If the agent stops without submitting it gets a nudge. On the last turn the code tries to force the submit with `tool_choice`, and falls back to instruction + nudge on models that don't support forcing (see *What broke* #5).

**4. Each specialist gets only the tools its job needs.**
The investigator can't escalate. The policy agent can only read policy. Only the communicator can write to the escalation queue.
*Why:* fewer tools means less chance of calling the wrong one, a smaller prompt per call, and the side effect (filing an escalation) sits behind a decision another agent already made.
*Cost:* more API calls and more latency, plus information can get lost in the handoff (the policy agent only knows what the investigator wrote down).

**5. Policy lives behind a tool, not in the system prompt.**
`lookup_policy(type)` returns only the policy for the exception types involved.
*Rule of thumb:* put something in the prompt when it's small and needed on every call. Put it behind a tool when it's large, per-case or owned by someone else (policy changes without a code change). Here the rules are per-type, and the multi path needs them in only one agent.

**6. Decision vs action is checked, not trusted.**
The orchestrator compares `escalate: true/false` with whether `create_escalation` was actually called and successful, and flags any mismatch (`escalation_consistent` in the eval).

**7. Tool errors go back to the model as data.**
Unknown IDs, bad arguments and tool exceptions come back as `is_error` tool results, so the agent can retry or correct itself instead of crashing the run.

## Eval

20 hand-labeled tickets (`data/tickets.json`): 6 late shipments, 4 inventory, 4 failed delivery, 4 quality, and 2 with two exception types. 6 should route multi and 10 should escalate. Edge cases are deliberate:
- **T03:** the customer asks "is it late?" and it isn't. The answer has to say "not late."
- **T05:** multi because of order value, but no escalation.
- **T13:** a key account (so multi) with no escalation.
- **T20:** gives only an order ID, so the agent has to call `get_order` to find the shipment.
- **T18, T19:** two problems in one ticket.

Labels follow from the written policies and the mock data. `tests/test_offline.py` re-derives every escalation and route label with an independent rule-based check, so a label error can't show up as a "model error."

Metrics per ticket: `types` (exact set), `route` (rule chose the expected route), `root_cause` (share of types diagnosed correctly), `escalate`, `consistent` (decision matches what was filed), `all_correct`. Cost: API calls, tokens and seconds, split by route.

```
python eval.py                 # normal routing
python eval.py --force single  # every ticket through one agent
python eval.py --force multi   # every ticket through the 3 specialists
```

## Results

<!-- RESULTS: from results/eval_*.json, one run per mode, model claude-sonnet-5-5 -->
One run per mode, 20 tickets, model `claude-sonnet-5-5`. Raw per-ticket output is in `results/`.

| Mode | all_correct | types | route | root_cause | escalate | consistent | avg API calls | avg tokens | avg seconds |
|---|---|---|---|---|---|---|---|---|---|
| routed (rule) | 20/20 | 100% | 100% | 100% | 100% | 100% | 4.65 | 12,883 | 16.5 |
| force single | 20/20 | 100% | n/a | 100% | 100% | 100% | 3.75 | 12,236 | 12.9 |
| force multi | 20/20 | 100% | n/a | 100% | 100% | 100% | 6.75 | 14,742 | 21.3 |

**How to read this:**
- **On this eval, the multi-agent pipeline buys nothing.** The single agent got every ticket right, including both two-problem tickets (T18, T19). The three-specialist pipeline made 1.8x the API calls, used 20% more tokens and took 65% longer, for the same score. The honest conclusion is that the routing rule currently spends the extra cost on 6 tickets without any measured benefit.
- **The eval is saturated, so it can't tell the architectures apart.** 100% across the board on 20 tickets whose labels follow directly from written rules means the tickets are too easy. Because the tools pre-compute `days_late`, `variance_pct` and the rest (design decision 2), each ticket reduces to "read the field, apply the rule." That's good for reliability, but it leaves little room for a weaker architecture to fail. Each mode was also run only once, so run-to-run variance is unmeasured.
- **What the score doesn't capture:** see *What broke* #6. A run scored 100% while silently dropping a required second escalation.

The next step is a harder eval, not a different architecture: ambiguous tickets, conflicting evidence, missing or wrong IDs, tool errors, and tickets that need three or more tools to combine. Then rerun the ablation with several runs per ticket.

## What broke (and what it taught me)

1. **The model had no idea what day it was.** In the step-1 prototype, a ticket asked "is SH-1001 late?" The model correctly said it couldn't tell without today's date. Then, in the same answer, it inferred lateness from a *different* shipment's delivery date and told the customer it was "almost certainly" late. That reasoning holds up, but the answer contradicts itself and would have gone to a customer. **Fix:** a fixed simulation date, and tools that return `days_late` so the model never does date math. **Lesson:** missing context doesn't make an LLM stop, it makes it improvise. Anything a decision depends on should be computed and handed to the model.
2. **`.env` is not magic.** The SDK reads `os.environ`, not `.env` files. `load_dotenv()` has to run first.
3. **API key scoping.** A key that isn't scoped to a workspace returns a 400 unless the request carries `anthropic-workspace-id`. Now supported via `ANTHROPIC_WORKSPACE_ID` in `.env`, sent as a default header.
4. **An unfair ablation, caught before running.** The first version gave the multi-path investigator extra root-cause guidance that the single resolver didn't have. Any accuracy gap for multi would have come from the prompt, not the architecture. Both now share one guidance block (`_root_cause_guide()`).

5. **Forced tool calls aren't supported by every model.** The first real eval run failed 60/60 ticket runs with `400: tool_choice type "tool" and "any" are not supported for this model`. The design leaned on forcing the submit tool, and the offline tests didn't catch it because the fake client accepted any `tool_choice`. **Fix:** `_create()` tries to force, and on that specific 400 it drops `tool_choice`, remembers the model can't be forced, and relies on an explicit "you must call submit_x" instruction plus a nudge if the model replies in prose. A new test fake rejects forcing the same way the real API does. **Lesson:** a fake client tests your logic, not the provider's contract. Model capabilities differ, so detect them and degrade gracefully instead of hard-coding them.

6. **A 100% score that hid a real bug: one escalation team per ticket.** In the T18 trace (late AND crushed), policy requires two escalations: logistics (late, key account) and quality (defect rate 45%). The output schema only allows one `escalation_team`, so only logistics was filed. The model noticed. Its internal note says *"The tool takes only one team, so the quality escalation (P2) for B-782 is NOT yet filed... Someone needs to file it separately."* The eval still scored the ticket correct, because it only checks the `escalate` true/false flag. **Two lessons:** (a) the schema is part of the design, and a schema that can't express the right answer forces a wrong one. (b) Metrics that are too coarse give false confidence, and reading the traces found what the summary hid. **Fix (next):** make `escalations` a list of `{team, priority}`, have the communicator file one per entry, and score the set of teams per ticket instead of a boolean.

## What I learned

- An "agent" is a while-loop: call the model, run the tools it asks for, send back results with matching `tool_use_id`s, and repeat until it stops. The API is stateless, so all memory is the `messages` list.
- The model never runs anything. It *asks*, and your code decides. That's where permissions, validation and logging belong.
- Tool descriptions are prompts. They decide when a tool gets called.
- Split work between LLM and code by asking "does this need language understanding, or does it need to be correct every time?"
- Multi-agent isn't automatically better. Here it cost 1.8x the API calls for the same score. The ablation flag (`--force single/multi`) is what made that visible.
- An eval at 100% is a warning sign, not a finish line. It usually means the eval is too easy, and reading the traces still finds real bugs.

## Run it

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Create `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
# only if your key is not workspace-scoped:
# ANTHROPIC_WORKSPACE_ID=wrkspc_...
# optional, default claude-sonnet-5-5:
# ROUTER_MODEL=claude-sonnet-5-5
```

```
python -m pytest -q          # offline tests, no API calls
python main.py T18           # one ticket with full trace
python main.py --text "SH-2004 is a week late" --customer CUST-ZEST
python eval.py               # full eval -> results/eval_routed.json
```

`step1_tool_loop.py` is the 60-line prototype of the raw tool-use loop that the rest is built on.

## Limitations / next steps

- Mock data with 20 tickets is a small eval. Each ticket is worth 5 points, so treat differences of one or two tickets as noise. Next: 100+ tickets and multiple runs per ticket to measure variance.
- Multi-team escalation (see *What broke* #6) isn't supported yet, and escalation is scored as a boolean rather than as the set of teams.
- Customer-message quality isn't scored. Next: an LLM-as-judge rubric (no promised dates without evidence, no blame, clear next step).
- The routing thresholds are hand-set. Next: tune them from the ablation results, i.e. route multi only for the ticket types where multi actually scores higher.
- No human-in-the-loop. In production, P1 escalations and customer messages for key accounts should be approved by a person before sending.
