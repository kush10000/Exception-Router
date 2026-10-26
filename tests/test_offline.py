"""Offline tests: no API key needed. Run with:  python -m pytest -q

1. Tools compute the right numbers.
2. Routing rule.
3. The agent loop, driven by a scripted fake client (protocol, error handling, forced submit).
4. Label check: the eval labels agree with an independent rule-based oracle over the mock data,
   so a wrong label can't silently show up as a model error.
"""
import json
from types import SimpleNamespace as NS

import pytest

from exception_router import agents, data, tools
from exception_router.agent import run_agent
from exception_router.routing import decide_route

T = tools.REGISTRY


# ---------------------------------------------------------------- 1. tools

def test_days_late():
    assert T["get_shipment"].fn("SH-2001")["days_late"] == 5      # promised 10-05, today 10-10
    assert T["get_shipment"].fn("SH-2003")["days_late"] == 0      # promised 10-12
    assert T["get_shipment"].fn("SH-2018")["days_late"] == 3      # delivered 10-08 vs promised 10-05
    assert T["get_shipment"].fn("SH-2003")["is_late"] is False


def test_inventory_and_quality_math():
    inv = T["get_inventory"].fn("SKU-LED-40", "WH-PUNE")
    assert inv["variance_units"] == -12 and inv["variance_pct"] == -6.0
    assert T["get_quality_inspection"].fn(batch_id="B-779")["defect_rate_pct"] == 15.0
    assert T["get_quality_inspection"].fn(shipment_id="SH-2014")["batch_id"] == "B-778"


def test_tools_return_errors_not_exceptions():
    assert "error" in T["get_shipment"].fn("SH-9999")
    assert "error" in T["get_inventory"].fn("SKU-X", "WH-X")
    assert "error" in T["get_quality_inspection"].fn()


def test_failed_attempts():
    a = T["get_delivery_attempts"].fn("SH-2010")
    assert a["failed_attempts"] == 3 and a["temperature_controlled"] is False


def test_schemas_are_valid_shapes():
    for t in T.values():
        s = t.schema
        assert s["input_schema"]["type"] == "object"
        assert set(s["input_schema"]["required"]) <= set(s["input_schema"]["properties"])


# -------------------------------------------------------------- 2. routing

def _tri(types, shipments=(), orders=(), batches=()):
    return {"exception_types": list(types), "shipment_ids": list(shipments), "order_ids": list(orders),
            "batch_ids": list(batches), "inventory_refs": [], "summary": ""}


def test_route_rules():
    nova = {"id": "x", "customer_id": "CUST-NOVA"}
    assert decide_route(nova, _tri(["late_shipment"], ["SH-2001"])).route == "single"
    assert decide_route(nova, _tri(["late_shipment"], ["SH-2005"])).route == "multi"          # high value
    assert decide_route({"id": "x", "customer_id": "CUST-ACME"}, _tri(["late_shipment"], ["SH-2002"])).route == "multi"
    assert decide_route(nova, _tri(["late_shipment", "quality_flag"], ["SH-2001"])).route == "multi"
    internal = {"id": "x", "customer_id": None}
    assert decide_route(internal, _tri(["inventory_mismatch"])).route == "single"
    # order found through an order ID only (T20) and through a batch ID only (T15)
    assert decide_route({"id": "x", "customer_id": None}, _tri(["late_shipment"], orders=["ORD-3005"])).route == "multi"
    assert decide_route({"id": "x", "customer_id": None}, _tri(["quality_flag"], batches=["B-778"])).route == "multi"


# ------------------------------------------------------- 3. loop with fake

class ForcingRejected(Exception):
    """Mimics anthropic.BadRequestError for models that don't support a forced tool_choice."""
    status_code = 400

    def __str__(self):
        return "tool_choice: type \"tool\" and \"any\" are not supported for this model."


@pytest.fixture(autouse=True)
def _reset_forcing_flag(monkeypatch):
    from exception_router import agent as agent_mod
    monkeypatch.setattr(agent_mod, "_FORCING_SUPPORTED", True)


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, script, reject_forcing=False):
        self.script = list(script)
        self.requests = []
        self.rejected = 0
        self.reject_forcing = reject_forcing
        self.messages = self

    def create(self, **kwargs):
        if self.reject_forcing and "tool_choice" in kwargs:
            self.rejected += 1
            raise ForcingRejected()
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        blocks = self.script.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


def tu(name, inp, id_):
    return NS(type="tool_use", name=name, input=inp, id=id_)


def text(t):
    return NS(type="text", text=t)


SUBMIT = {"name": "submit_x", "description": "d",
          "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}


def test_loop_runs_tools_and_matches_ids():
    fake = FakeClient([
        [tu("get_shipment", {"shipment_id": "SH-2001"}, "t1"), tu("get_shipment", {"shipment_id": "SH-2003"}, "t2")],
        [tu("submit_x", {"a": "done"}, "t3")],
    ])
    run = run_agent(fake, agent="a", system="s", tools=tools.get("get_shipment"), submit=SUBMIT, user_message="u")
    assert run.output == {"a": "done"} and run.api_calls == 2 and len(run.tool_calls) == 2
    results = fake.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]           # parallel calls answered in one message
    assert json.loads(results[0]["content"])["days_late"] == 5


def test_loop_reports_errors_to_model():
    fake = FakeClient([
        [tu("get_shipment", {"wrong_arg": 1}, "t1"), tu("no_such_tool", {}, "t2")],
        [tu("submit_x", {"a": "ok"}, "t3")],
    ])
    run = run_agent(fake, agent="a", system="s", tools=tools.get("get_shipment"), submit=SUBMIT, user_message="u")
    results = fake.requests[1]["messages"][-1]["content"]
    assert all(r["is_error"] for r in results)
    assert run.output == {"a": "ok"}


def test_loop_nudges_then_forces_submit():
    fake = FakeClient([[text("thinking...")], [text("still prose")], [tu("submit_x", {"a": "forced"}, "t1")]])
    run = run_agent(fake, agent="a", system="s", tools=tools.get("get_shipment"), submit=SUBMIT,
                    user_message="u", max_turns=3)
    assert "tool_choice" not in fake.requests[0]
    assert fake.requests[2]["tool_choice"] == {"type": "tool", "name": "submit_x"}
    assert run.forced_submit and run.output == {"a": "forced"}


def test_triage_is_one_forced_call():
    fake = FakeClient([[tu("submit_triage", _tri(["late_shipment"], ["SH-2001"]), "t1")]])
    run = agents.triage(fake, {"id": "T", "customer_id": "CUST-NOVA", "text": "x"})
    assert run.api_calls == 1 and fake.requests[0]["tool_choice"]["name"] == "submit_triage"


# ----------------------------------------------------------- 4. label check

def oracle(ticket):
    """Independent re-derivation of escalate + route from the written policy rules and the data."""
    exp = ticket["expected"]
    cust = data.customer(ticket["customer_id"])
    key = bool(cust and cust["tier"] == "key_account")
    esc = False
    ids = ticket["text"]
    for etype in exp["exception_types"]:
        if etype == "late_shipment":
            sid = next((s for s in data.world()["shipments"] if s in ids), None)
            if sid is None:  # order-only ticket
                oid = next(o for o in data.world()["orders"] if o in ids)
                sid = data.order(oid)["shipment_ids"][0]
            dl = T["get_shipment"].fn(sid)["days_late"]
            esc |= (dl >= 3 and key) or dl >= 7
        elif etype == "failed_delivery":
            sid = next(s for s in data.world()["shipments"] if s in ids)
            a = T["get_delivery_attempts"].fn(sid)
            esc |= a["failed_attempts"] >= 3 or a["temperature_controlled"]
        elif etype == "inventory_mismatch":
            k = next(k for k in data.world()["inventory"] if k.split("@")[0] in ids)
            esc |= abs(T["get_inventory"].fn(*k.split("@"))["variance_pct"]) >= 5
        elif etype == "quality_flag":
            b = next(b for b in data.world()["quality_inspections"] if b in ids)
            q = T["get_quality_inspection"].fn(batch_id=b)
            esc |= q["severity"] == "critical" or q["defect_rate_pct"] >= 10
    return esc


@pytest.mark.parametrize("ticket", data.load_tickets(), ids=lambda t: t["id"])
def test_labels_match_policy(ticket):
    exp = ticket["expected"]
    assert oracle(ticket) == exp["escalate"]
    sids = [s for s in data.world()["shipments"] if s in ticket["text"]]
    oids = [o for o in data.world()["orders"] if o in ticket["text"]]
    bids = [b for b in data.world()["quality_inspections"] if b in ticket["text"]]
    assert decide_route(ticket, _tri(exp["exception_types"], sids, oids, bids)).route == exp["route"]
    for etype, rc in exp["root_causes"].items():
        assert rc in agents.ROOT_CAUSES[etype]


def test_eval_set_balance():
    ts = data.load_tickets()
    assert len(ts) == 20
    assert sum(t["expected"]["route"] == "multi" for t in ts) >= 6
    assert sum(t["expected"]["escalate"] for t in ts) >= 8


# ------------------------------------------------- 5. orchestrator plumbing

def _finding(t, rc):
    return {"exception_type": t, "root_cause": rc, "evidence": "e", "confidence": "high"}


DECISION = {"escalate": True, "escalation_team": "logistics", "priority": "P1",
            "policy_rules_applied": ["r"], "required_actions": ["a"]}
COMMS = {"customer_message": "m", "internal_note": "n", "escalation_id": "ESC-1"}


def test_orchestrator_multi_path(tmp_path, monkeypatch):
    from exception_router import orchestrator
    monkeypatch.setattr(tools, "OUTBOX_DIR", tmp_path)
    ticket = next(t for t in data.load_tickets() if t["id"] == "T02")
    fake = FakeClient([
        [tu("submit_triage", _tri(["late_shipment"], ["SH-2002"]), "a")],
        [tu("get_shipment", {"shipment_id": "SH-2002"}, "b")],
        [tu("submit_findings", {"findings": [_finding("late_shipment", "warehouse_dispatch_delay")],
                                "key_facts": {"days_late": 4}}, "c")],
        [tu("lookup_policy", {"exception_type": "late_shipment"}, "d")],
        [tu("submit_policy_decision", DECISION, "e")],
        [tu("create_escalation", {"ticket_id": "T02", "team": "logistics", "priority": "P1", "summary": "s"}, "f")],
        [tu("submit_communications", COMMS, "g")],
    ])
    res = orchestrator.handle_ticket(fake, ticket)
    assert res["route"] == "multi" and res["root_causes"] == {"late_shipment": "warehouse_dispatch_delay"}
    assert res["escalations_filed"] == 1 and res["escalation_consistent"]
    assert [a["agent"] for a in res["stats"]["agents"]] == ["triage", "investigator", "policy", "communicator"]
    assert (tmp_path / "escalations.jsonl").exists()
    # the policy specialist only ever sees lookup_policy + its submit tool
    assert [t["name"] for t in fake.requests[3]["tools"]] == ["lookup_policy", "submit_policy_decision"]


def test_orchestrator_single_path_flags_inconsistency():
    from exception_router import orchestrator
    ticket = next(t for t in data.load_tickets() if t["id"] == "T01")
    fake = FakeClient([
        [tu("submit_triage", _tri(["late_shipment"], ["SH-2001"]), "a")],
        [tu("submit_resolution", {"findings": [_finding("late_shipment", "carrier_hub_disruption")],
                                  **DECISION, **COMMS}, "b")],   # says escalate, never filed
    ])
    res = orchestrator.handle_ticket(fake, ticket)
    assert res["route"] == "single" and res["escalations_filed"] == 0
    assert res["escalation_consistent"] is False



# ---------------------------------- 6. models that reject forced tool_choice

def test_fallback_when_model_rejects_forcing():
    from exception_router import agent as agent_mod
    fake = FakeClient([[text("prose instead of a tool call")],
                       [tu("submit_triage", _tri(["late_shipment"], ["SH-2001"]), "t1")]],
                      reject_forcing=True)
    run = agents.triage(fake, {"id": "T", "customer_id": "CUST-NOVA", "text": "x"})
    assert fake.rejected == 1                              # tried forcing once, got the 400
    assert agent_mod._FORCING_SUPPORTED is False           # remembered for the rest of the process
    assert all("tool_choice" not in r for r in fake.requests)
    assert "Do not reply in text" in fake.requests[1]["messages"][-1]["content"]   # nudged
    assert run.output["exception_types"] == ["late_shipment"] and run.forced_submit is False


def test_no_forcing_and_no_submit_raises():
    from exception_router.agent import AgentError
    fake = FakeClient([[text("a")], [text("b")]], reject_forcing=True)
    with pytest.raises(AgentError):
        run_agent(fake, agent="a", system="s", tools=tools.get("get_shipment"), submit=SUBMIT,
                  user_message="u", max_turns=2)


def test_unrelated_errors_are_not_swallowed():
    class Boom(Exception):
        status_code = 500
    class C:
        messages = None
        def create(self, **kw):
            raise Boom("server error")
    c = C(); c.messages = c
    with pytest.raises(Boom):
        agents.triage(c, {"id": "T", "customer_id": None, "text": "x"})
