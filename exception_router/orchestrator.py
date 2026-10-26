"""Orchestrator: triage -> route -> single resolver OR investigator -> policy -> communicator."""
import time

from . import agents
from .agent import AgentRun
from .routing import decide_route


def _summarise(runs: list[AgentRun]) -> dict:
    return {
        "api_calls": sum(r.api_calls for r in runs),
        "tool_calls": sum(len(r.tool_calls) for r in runs),
        "tool_errors": sum(sum(c["error"] for c in r.tool_calls) for r in runs),
        "input_tokens": sum(r.input_tokens for r in runs),
        "output_tokens": sum(r.output_tokens for r in runs),
        "forced_submits": sum(r.forced_submit for r in runs),
        "agents": [{"agent": r.agent, "api_calls": r.api_calls, "seconds": r.seconds,
                    "tools": [c["tool"] for c in r.tool_calls]} for r in runs],
    }


def handle_ticket(client, ticket: dict, force_route: str | None = None, verbose: bool = False) -> dict:
    t0 = time.time()
    runs: list[AgentRun] = []

    tri_run = agents.triage(client, ticket, verbose=verbose)
    runs.append(tri_run)
    tri = tri_run.output

    decision = decide_route(ticket, tri)
    route = force_route or decision.route
    if verbose:
        forced = f" (FORCED, rule said {decision.route})" if force_route and force_route != decision.route else ""
        print(f"  route: {route}{forced}  reasons: {decision.reasons}")

    if route == "single":
        r = agents.resolver(client, ticket, tri, verbose=verbose)
        runs.append(r)
        out = r.output
        findings = out.get("findings", [])
        policy = {k: out.get(k) for k in agents.DECISION_PROPS}
        comms = {k: out.get(k) for k in agents.COMMS_PROPS}
    else:
        inv = agents.investigator(client, ticket, tri, verbose=verbose)
        runs.append(inv)
        pol = agents.policy_specialist(client, ticket, tri, inv.output, verbose=verbose)
        runs.append(pol)
        com = agents.communicator(client, ticket, inv.output, pol.output, verbose=verbose)
        runs.append(com)
        findings, policy, comms = inv.output.get("findings", []), pol.output, com.output

    escalations_filed = sum(c["tool"] == "create_escalation" and not c["error"]
                            for r in runs for c in r.tool_calls)
    return {
        "ticket_id": ticket["id"],
        "triage": tri,
        "route": route,
        "rule_route": decision.route,
        "route_reasons": decision.reasons,
        "root_causes": {f["exception_type"]: f["root_cause"] for f in findings},
        "findings": findings,
        "decision": policy,
        "communications": comms,
        "escalations_filed": escalations_filed,
        # The decision said escalate but nothing was filed, or the reverse. Caught here, not left to trust.
        "escalation_consistent": bool(policy.get("escalate")) == (escalations_filed > 0),
        "stats": {**_summarise(runs), "seconds": round(time.time() - t0, 2)},
    }
