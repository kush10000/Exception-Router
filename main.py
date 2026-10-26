"""CLI: route one ticket and show the trace.

  python main.py T02                      # a ticket from data/tickets.json
  python main.py T02 --force single       # override the routing rule
  python main.py --text "SH-2004 is late" --customer CUST-ZEST
  python main.py T18 --json               # full JSON result
"""
import argparse
import json
import sys

from exception_router.config import make_client
from exception_router.data import load_tickets
from exception_router.orchestrator import handle_ticket


def main():
    ap = argparse.ArgumentParser(description="Supply-chain exception router")
    ap.add_argument("ticket_id", nargs="?", help="Ticket ID from data/tickets.json")
    ap.add_argument("--text", help="Free-text ticket instead of a ticket ID")
    ap.add_argument("--customer", help="Customer ID for --text tickets, e.g. CUST-ACME")
    ap.add_argument("--force", choices=["single", "multi"], help="Force a route")
    ap.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = ap.parse_args()

    if args.text:
        ticket = {"id": "ADHOC", "customer_id": args.customer, "source": "cli", "text": args.text}
    elif args.ticket_id:
        ticket = next((t for t in load_tickets() if t["id"] == args.ticket_id), None)
        if not ticket:
            sys.exit(f"Unknown ticket {args.ticket_id}")
    else:
        ap.error("give a ticket ID or --text")

    print(f"\n=== {ticket['id']}: {ticket['text']}\n")
    res = handle_ticket(make_client(), ticket, force_route=args.force, verbose=True)

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    d, c, s = res["decision"], res["communications"], res["stats"]
    print(f"\nTypes:       {res['triage']['exception_types']}")
    print(f"Route:       {res['route']}  ({'; '.join(res['route_reasons'])})")
    for f in res["findings"]:
        print(f"Root cause:  {f['exception_type']} -> {f['root_cause']} [{f['confidence']}]  {f['evidence']}")
    print(f"Escalate:    {d['escalate']}  team={d['escalation_team']} priority={d['priority']}"
          f"  filed={res['escalations_filed']}")
    for rule in d.get("policy_rules_applied", []):
        print(f"  rule:      {rule}")
    for a in d.get("required_actions", []):
        print(f"  action:    {a}")
    if c.get("customer_message"):
        print(f"\nCustomer message:\n{c['customer_message']}")
    print(f"\nInternal note: {c.get('internal_note')}")
    print(f"\nCost: {s['api_calls']} API calls, {s['tool_calls']} tool calls, "
          f"{s['input_tokens']} in / {s['output_tokens']} out tokens, {s['seconds']}s")
    if not res["escalation_consistent"]:
        print("WARNING: escalation decision and filed escalations disagree")


if __name__ == "__main__":
    main()
