"""Single vs multi routing: the LLM extracts, deterministic code decides.

Why: the routing rule is a business rule (who gets the expensive, more careful pipeline). Business rules
belong in code where they are auditable, testable and free. The LLM's job is the part code can't do:
reading free text to find exception types and IDs.

multi if any of:
  * more than one exception type in the ticket
  * customer is a key account
  * any linked order is worth >= high_value_threshold_inr
"""
from dataclasses import dataclass

from . import data


@dataclass
class RouteDecision:
    route: str               # "single" | "multi"
    reasons: list[str]
    customer_tier: str | None
    max_order_value_inr: int | None


def resolve_orders(ticket: dict, tri: dict) -> list[dict]:
    order_ids = list(tri.get("order_ids") or [])
    for sid in tri.get("shipment_ids") or []:
        s = data.shipment(sid)
        if s and s["order_id"] not in order_ids:
            order_ids.append(s["order_id"])
    for bid in tri.get("batch_ids") or []:
        insp = data.world()["quality_inspections"].get(bid)
        if insp:
            s = data.shipment(insp["shipment_id"])
            if s and s["order_id"] not in order_ids:
                order_ids.append(s["order_id"])
    return [o for o in (data.order(i) for i in order_ids) if o]


def decide_route(ticket: dict, tri: dict) -> RouteDecision:
    orders = resolve_orders(ticket, tri)
    cust = data.customer(ticket.get("customer_id"))
    if not cust and orders:
        cust = data.customer(orders[0]["customer_id"])
    tier = cust["tier"] if cust else None
    max_value = max((o["value_inr"] for o in orders), default=None)
    threshold = data.world()["high_value_threshold_inr"]

    reasons = []
    types = tri.get("exception_types") or []
    if len(types) > 1:
        reasons.append(f"{len(types)} exception types ({', '.join(types)})")
    if tier == "key_account":
        reasons.append(f"key account ({cust['name']})")
    if max_value is not None and max_value >= threshold:
        reasons.append(f"high-value order (INR {max_value:,} >= {threshold:,})")

    if reasons:
        return RouteDecision("multi", reasons, tier, max_value)
    return RouteDecision("single", ["one exception type, standard/internal customer, order below threshold"],
                         tier, max_value)
