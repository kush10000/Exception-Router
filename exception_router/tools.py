"""Tools the agents can call.

Design rules (each one came from a failure in step1_tool_loop.py or from the eval):
  * Tools do the arithmetic. days_late, variance_pct, defect_rate_pct and failed_attempts are
    computed here against the simulation date, so the model never does date math or percentages.
  * Tools return errors as data ({"error": ...}) rather than raising, so the model can recover.
  * Each agent only gets the tools its job needs (see TOOLSETS).
"""
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Callable

from . import data
from .config import OUTBOX_DIR


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., dict]

    @property
    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict, required: list[str] | None = None):
    def wrap(fn):
        REGISTRY[name] = Tool(
            name=name,
            description=description,
            input_schema={"type": "object", "properties": properties, "required": required or []},
            fn=fn,
        )
        return fn
    return wrap


def _days_late(s: dict) -> int:
    promised = date.fromisoformat(s["promised_date"])
    if s.get("delivered_date"):
        return max(0, (date.fromisoformat(s["delivered_date"]) - promised).days)
    return max(0, (data.today() - promised).days)


# ---------------------------------------------------------------- read tools

@tool("get_order",
      "Look up an order: customer, customer tier (key_account/standard), order value in INR, and its "
      "shipments with status and days_late. Use when a ticket gives an order ID instead of a shipment ID.",
      {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-3001"}}, ["order_id"])
def get_order(order_id: str) -> dict:
    o = data.order(order_id)
    if not o:
        return {"error": f"No order with id {order_id}"}
    shipments = []
    for sid in o["shipment_ids"]:
        s = data.shipment(sid)
        shipments.append({"shipment_id": sid, "status": s["status"], "promised_date": s["promised_date"],
                          "days_late": _days_late(s)})
    return {**o, "customer": data.customer(o["customer_id"]), "shipments": shipments}


@tool("get_shipment",
      "Look up a shipment: carrier, route, status, promised and delivered dates, scan events, and "
      "days_late (already computed against today's date, 0 means not late). Use for any ticket that "
      "mentions a shipment ID.",
      {"shipment_id": {"type": "string", "description": "Shipment ID, e.g. SH-2001"}}, ["shipment_id"])
def get_shipment(shipment_id: str) -> dict:
    s = data.shipment(shipment_id)
    if not s:
        return {"error": f"No shipment with id {shipment_id}"}
    o = data.order(s["order_id"])
    dl = _days_late(s)
    out = {k: v for k, v in s.items() if k != "delivery_attempts"}
    out.update({
        "today": data.today().isoformat(),
        "days_late": dl,
        "is_late": dl > 0,
        "last_event": s["events"][-1] if s["events"] else None,
        "customer": data.customer(o["customer_id"]) if o else None,
        "order_value_inr": o["value_inr"] if o else None,
    })
    return out


@tool("get_delivery_attempts",
      "Get the delivery attempts for a shipment with reason codes, the number of failed attempts, "
      "whether the address is verified and whether the goods are temperature-controlled.",
      {"shipment_id": {"type": "string"}}, ["shipment_id"])
def get_delivery_attempts(shipment_id: str) -> dict:
    s = data.shipment(shipment_id)
    if not s:
        return {"error": f"No shipment with id {shipment_id}"}
    attempts = s["delivery_attempts"]
    return {
        "shipment_id": shipment_id,
        "status": s["status"],
        "attempts": attempts,
        "failed_attempts": sum(1 for a in attempts if a["outcome"] == "failed"),
        "address_verified": s["address_verified"],
        "temperature_controlled": s["temperature_controlled"],
    }


@tool("get_carrier_status",
      "Get a carrier's 30-day on-time rate and any declared network disruptions (hub, date range, reason). "
      "Use to tell a carrier-side disruption apart from an ordinary carrier delay.",
      {"carrier": {"type": "string", "description": "Carrier name exactly as in the shipment, e.g. BlueDart"}},
      ["carrier"])
def get_carrier_status(carrier: str) -> dict:
    c = data.world()["carriers"].get(carrier)
    if not c:
        return {"error": f"Unknown carrier {carrier}. Known: {sorted(data.world()['carriers'])}"}
    return {"carrier": carrier, **c, "today": data.today().isoformat()}


@tool("get_inventory",
      "Get stock for a SKU at a warehouse: system quantity, physical count, variance_units and "
      "variance_pct (already computed), last cycle count, recent stock movements and notes.",
      {"sku": {"type": "string", "description": "e.g. SKU-LED-40"},
       "warehouse": {"type": "string", "description": "e.g. WH-PUNE"}},
      ["sku", "warehouse"])
def get_inventory(sku: str, warehouse: str) -> dict:
    rec = data.world()["inventory"].get(f"{sku}@{warehouse}")
    if not rec:
        known = sorted(data.world()["inventory"])
        return {"error": f"No inventory record for {sku} at {warehouse}. Known records: {known}"}
    var = rec["physical_qty"] - rec["system_qty"]
    return {"sku": sku, "warehouse": warehouse, **rec,
            "variance_units": var,
            "variance_pct": round(100 * var / rec["system_qty"], 2)}


@tool("get_quality_inspection",
      "Get the receiving quality inspection for a batch (or for a shipment if you only have the shipment ID): "
      "supplier, defect type, severity (minor/major/critical), defect_rate_pct (already computed) and inspector notes.",
      {"batch_id": {"type": "string", "description": "e.g. B-778"},
       "shipment_id": {"type": "string", "description": "Use if no batch ID is known"}})
def get_quality_inspection(batch_id: str | None = None, shipment_id: str | None = None) -> dict:
    insp = data.world()["quality_inspections"]
    if batch_id:
        rec = insp.get(batch_id)
    elif shipment_id:
        batch_id, rec = next(((b, r) for b, r in insp.items() if r["shipment_id"] == shipment_id), (None, None))
    else:
        return {"error": "Provide batch_id or shipment_id"}
    if not rec:
        return {"error": f"No inspection found for batch_id={batch_id} shipment_id={shipment_id}"}
    return {"batch_id": batch_id, **rec,
            "defect_rate_pct": round(100 * rec["defective_units"] / rec["inspected_units"], 2)}


@tool("lookup_policy",
      "Get the company policy for an exception type: root-cause definitions, escalation rules, priority "
      "and required actions. Always check the policy before deciding whether to escalate.",
      {"exception_type": {"type": "string",
                          "enum": ["late_shipment", "inventory_mismatch", "failed_delivery", "quality_flag"]}},
      ["exception_type"])
def lookup_policy(exception_type: str) -> dict:
    p = data.policies()
    if exception_type not in p:
        return {"error": f"No policy for {exception_type}"}
    return {"exception_type": exception_type, "policy": p[exception_type], "general": p["general"]}


# --------------------------------------------------------------- write tool

_outbox_lock = threading.Lock()


@tool("create_escalation",
      "File an escalation to an internal team. This writes a real record to the escalation queue, so only "
      "call it when the policy says to escalate. Returns the escalation ID.",
      {"ticket_id": {"type": "string"},
       "team": {"type": "string", "enum": ["logistics", "warehouse_ops", "quality", "customer_success"]},
       "priority": {"type": "string", "enum": ["P1", "P2"]},
       "summary": {"type": "string", "description": "2-4 sentences: what happened, root cause, evidence, ask."}},
      ["ticket_id", "team", "priority", "summary"])
def create_escalation(ticket_id: str, team: str, priority: str, summary: str) -> dict:
    esc_id = "ESC-" + uuid.uuid4().hex[:6].upper()
    rec = {"escalation_id": esc_id, "ticket_id": ticket_id, "team": team, "priority": priority, "summary": summary}
    OUTBOX_DIR.mkdir(exist_ok=True)
    with _outbox_lock, open(OUTBOX_DIR / "escalations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"status": "filed", "escalation_id": esc_id}


# ----------------------------------------------------------- tool subsets

INVESTIGATION_TOOLS = {
    "late_shipment": ["get_order", "get_shipment", "get_carrier_status"],
    "failed_delivery": ["get_order", "get_shipment", "get_delivery_attempts", "get_carrier_status"],
    "inventory_mismatch": ["get_inventory"],
    "quality_flag": ["get_order", "get_shipment", "get_quality_inspection"],
}


def investigation_tools(exception_types: list[str]) -> list[Tool]:
    names: list[str] = []
    for t in exception_types:
        for n in INVESTIGATION_TOOLS.get(t, []):
            if n not in names:
                names.append(n)
    return [REGISTRY[n] for n in names]


def get(*names: str) -> list[Tool]:
    return [REGISTRY[n] for n in names]
