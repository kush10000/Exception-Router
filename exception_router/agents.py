"""Agent definitions: system prompts + tool subsets + submit schemas.

  triage        - 1 forced call, no tools. Reads the ticket, extracts types and IDs.
  resolver      - SINGLE-agent path: investigates, applies policy and drafts replies, all in one loop.
  investigator  - MULTI path, specialist 1: root cause from evidence. Read-only tools.
  policy        - MULTI path, specialist 2: applies written policy to the findings. Only lookup_policy.
  communicator  - MULTI path, specialist 3: files the escalation and drafts the messages. Only create_escalation.
"""
from . import data, tools
from .agent import AgentRun, run_agent

EXCEPTION_TYPES = ["late_shipment", "inventory_mismatch", "failed_delivery", "quality_flag"]
ROOT_CAUSES = {
    "late_shipment": ["carrier_hub_disruption", "carrier_delay", "warehouse_dispatch_delay", "not_late"],
    "inventory_mismatch": ["duplicate_posting", "unrecorded_pick", "damaged_not_written_off", "theft_or_loss_suspected"],
    "failed_delivery": ["customer_unavailable", "address_issue", "receiver_refused", "carrier_operational_failure"],
    "quality_flag": ["supplier_defect", "damaged_in_transit"],
}
ALL_ROOT_CAUSES = sorted({rc for v in ROOT_CAUSES.values() for rc in v})


def _ctx() -> str:
    return (f"Today's date is {data.today().isoformat()}. Tools return computed fields (days_late, variance_pct, "
            f"defect_rate_pct, failed_attempts); use them and never do date or percentage arithmetic yourself. "
            f"Only state facts that a tool returned.")


HOW_TO_DIAGNOSE = """How to tell root causes apart:
- late_shipment: check get_shipment (days_late, events). If days_late is 0, the root cause is not_late. If the
  shipment never left our warehouse, it is warehouse_dispatch_delay. Otherwise check get_carrier_status: a declared
  disruption at the hub where the shipment is stuck means carrier_hub_disruption; no disruption means carrier_delay.
- failed_delivery: use the reason codes and address_verified from get_delivery_attempts.
- inventory_mismatch: compare the movements and notes from get_inventory against the variance.
- quality_flag: inspector notes say whether the defect existed before shipping (supplier_defect) or the goods were
  fine at origin and damaged on the way (damaged_in_transit)."""


def _root_cause_guide() -> str:
    """Shared by the single resolver and the multi investigator, so the single-vs-multi comparison is fair."""
    allowed = "\n".join(f"- {t}: {', '.join(rcs)}" for t, rcs in ROOT_CAUSES.items())
    return f"{allowed}\n\n{HOW_TO_DIAGNOSE}"


FINDINGS_SCHEMA = {
    "type": "array",
    "description": "One finding per exception type in the ticket.",
    "items": {
        "type": "object",
        "properties": {
            "exception_type": {"type": "string", "enum": EXCEPTION_TYPES},
            "root_cause": {"type": "string", "enum": ALL_ROOT_CAUSES},
            "evidence": {"type": "string", "description": "The specific tool facts that support the root cause."},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["exception_type", "root_cause", "evidence", "confidence"],
    },
}

DECISION_PROPS = {
    "escalate": {"type": "boolean"},
    "escalation_team": {"type": "string", "enum": ["logistics", "warehouse_ops", "quality", "customer_success", "none"]},
    "priority": {"type": "string", "enum": ["P1", "P2", "none"]},
    "policy_rules_applied": {"type": "array", "items": {"type": "string"},
                             "description": "Quote the policy rule(s) that decided escalation, with the numbers."},
    "required_actions": {"type": "array", "items": {"type": "string"}},
}

COMMS_PROPS = {
    "customer_message": {"type": "string",
                         "description": "Message to the customer. Empty string for internal tickets with no customer."},
    "internal_note": {"type": "string", "description": "Short note for the ops team."},
    "escalation_id": {"type": "string", "description": "ID returned by create_escalation, or empty string."},
}


# ------------------------------------------------------------------ triage

TRIAGE_SYSTEM = f"""You triage supply-chain exception tickets. Read the ticket and extract structured fields.

Exception types:
- late_shipment: the complaint is about a shipment being late or about its ETA (including "is it going to be late?").
- failed_delivery: the complaint is that a delivery attempt failed or the goods came back undelivered.
- inventory_mismatch: system stock and physical count disagree.
- quality_flag: received goods are damaged or defective.

Tag more than one type only if the ticket raises distinct problems (e.g. late AND damaged). A failed delivery is
not also a late shipment unless the customer separately complains about lateness.
Copy IDs exactly as written. Do not invent IDs."""

TRIAGE_TOOL = {
    "name": "submit_triage",
    "description": "Submit the triage result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "exception_types": {"type": "array", "items": {"type": "string", "enum": EXCEPTION_TYPES}, "minItems": 1},
            "shipment_ids": {"type": "array", "items": {"type": "string"}},
            "order_ids": {"type": "array", "items": {"type": "string"}},
            "batch_ids": {"type": "array", "items": {"type": "string"}},
            "inventory_refs": {"type": "array", "description": "SKU + warehouse pairs mentioned.",
                               "items": {"type": "object",
                                         "properties": {"sku": {"type": "string"}, "warehouse": {"type": "string"}},
                                         "required": ["sku", "warehouse"]}},
            "summary": {"type": "string", "description": "One sentence."},
        },
        "required": ["exception_types", "shipment_ids", "order_ids", "batch_ids", "inventory_refs", "summary"],
    },
}


def ticket_block(ticket: dict) -> str:
    cust = data.customer(ticket.get("customer_id"))
    who = f"{cust['name']} ({cust['customer_id']}, tier={cust['tier']})" if cust else "internal (no customer)"
    return f"Ticket {ticket['id']} | source={ticket.get('source')} | from: {who}\n\n{ticket['text']}"


def triage(client, ticket: dict, verbose=False) -> AgentRun:
    return run_agent(client, agent="triage", system=TRIAGE_SYSTEM, tools=[], submit=TRIAGE_TOOL,
                     user_message=ticket_block(ticket), max_turns=3, verbose=verbose)


def _triage_block(tri: dict) -> str:
    return "Triage result:\n" + "\n".join(f"- {k}: {v}" for k, v in tri.items() if v)


# -------------------------------------------------------- single: resolver

RESOLVER_SYSTEM = """You are a supply-chain exception resolver. You handle the whole ticket alone:
1. Investigate with the tools until you know the root cause of each exception.
2. Look up the policy for each exception type and decide escalation strictly by the policy rules.
3. If the policy says escalate, call create_escalation once. Otherwise do not.
4. Draft the customer message (empty for internal tickets) and a short internal note.
5. Call submit_resolution.

Root causes allowed per type:
{rcs}

{ctx}"""

RESOLVER_TOOL = {
    "name": "submit_resolution",
    "description": "Submit the final resolution for the ticket.",
    "input_schema": {"type": "object",
                     "properties": {"findings": FINDINGS_SCHEMA, **DECISION_PROPS, **COMMS_PROPS},
                     "required": ["findings", *DECISION_PROPS, *COMMS_PROPS]},
}


def resolver(client, ticket, tri, verbose=False) -> AgentRun:
    ts = tools.investigation_tools(tri["exception_types"]) + tools.get("lookup_policy", "create_escalation")
    return run_agent(client, agent="resolver",
                     system=RESOLVER_SYSTEM.format(rcs=_root_cause_guide(), ctx=_ctx()),
                     tools=ts, submit=RESOLVER_TOOL,
                     user_message=f"{ticket_block(ticket)}\n\n{_triage_block(tri)}", max_turns=10, verbose=verbose)


# ---------------------------------------------------- multi: specialists

INVESTIGATOR_SYSTEM = """You are the investigation specialist. Your only job is root cause.
Gather evidence with the tools for every exception type in the triage, then submit one finding per type.
Do not decide escalation and do not write customer messages. Other specialists do that.

Root causes allowed per type:
{rcs}

{ctx}"""

INVESTIGATOR_TOOL = {
    "name": "submit_findings",
    "description": "Submit root-cause findings, one per exception type, plus key facts the policy specialist needs.",
    "input_schema": {"type": "object",
                     "properties": {"findings": FINDINGS_SCHEMA,
                                    "key_facts": {"type": "object",
                                                  "description": "Numbers the policy needs, e.g. days_late, customer_tier, "
                                                                 "variance_pct, failed_attempts, temperature_controlled, "
                                                                 "severity, defect_rate_pct."}},
                     "required": ["findings", "key_facts"]},
}


def investigator(client, ticket, tri, verbose=False) -> AgentRun:
    return run_agent(client, agent="investigator",
                     system=INVESTIGATOR_SYSTEM.format(rcs=_root_cause_guide(), ctx=_ctx()),
                     tools=tools.investigation_tools(tri["exception_types"]), submit=INVESTIGATOR_TOOL,
                     user_message=f"{ticket_block(ticket)}\n\n{_triage_block(tri)}", max_turns=10, verbose=verbose)


POLICY_SYSTEM = """You are the policy specialist. You receive investigation findings and apply company policy.
Look up the policy for every exception type involved. Decide escalation strictly by the written rules and the
numbers in the findings: no judgment calls beyond the policy text. If any exception type requires escalation,
escalate=true. Quote the deciding rule with the numbers in policy_rules_applied.

{ctx}"""

POLICY_TOOL = {
    "name": "submit_policy_decision",
    "description": "Submit the escalation decision and required actions.",
    "input_schema": {"type": "object", "properties": DECISION_PROPS, "required": list(DECISION_PROPS)},
}


def policy_specialist(client, ticket, tri, findings: dict, verbose=False) -> AgentRun:
    msg = (f"{ticket_block(ticket)}\n\n{_triage_block(tri)}\n\n"
           f"Investigation findings:\n{findings}")
    return run_agent(client, agent="policy", system=POLICY_SYSTEM.format(ctx=_ctx()),
                     tools=tools.get("lookup_policy"), submit=POLICY_TOOL,
                     user_message=msg, max_turns=6, verbose=verbose)


COMMS_SYSTEM = """You are the communications specialist. You receive the findings and the policy decision.
1. If the decision says escalate=true, call create_escalation exactly once with the given team and priority.
   If escalate=false, do not call it.
2. Write the customer message: apologetic where appropriate, factual, says what happens next, and promises no
   date that is not in the findings. For internal tickets with no customer, customer_message is "".
3. Write a short internal note. Then call submit_communications.

{ctx}"""

COMMS_TOOL = {
    "name": "submit_communications",
    "description": "Submit the drafted messages.",
    "input_schema": {"type": "object", "properties": COMMS_PROPS, "required": list(COMMS_PROPS)},
}


def communicator(client, ticket, findings: dict, decision: dict, verbose=False) -> AgentRun:
    msg = (f"{ticket_block(ticket)}\n\nFindings:\n{findings}\n\nPolicy decision:\n{decision}")
    return run_agent(client, agent="communicator", system=COMMS_SYSTEM.format(ctx=_ctx()),
                     tools=tools.get("create_escalation"), submit=COMMS_TOOL,
                     user_message=msg, max_turns=4, verbose=verbose)
