"""Mock data layer. Stands in for the client's TMS / WMS / QMS APIs."""
import json
from datetime import date
from functools import lru_cache

from .config import DATA_DIR


@lru_cache(maxsize=1)
def world() -> dict:
    return json.loads((DATA_DIR / "world.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def policies() -> dict:
    return json.loads((DATA_DIR / "policies.json").read_text(encoding="utf-8"))


def load_tickets() -> list[dict]:
    return json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))


def today() -> date:
    return date.fromisoformat(world()["today"])


def customer(customer_id: str | None) -> dict | None:
    if not customer_id:
        return None
    c = world()["customers"].get(customer_id)
    return {"customer_id": customer_id, **c} if c else None


def order(order_id: str) -> dict | None:
    o = world()["orders"].get(order_id)
    return {"order_id": order_id, **o} if o else None


def shipment(shipment_id: str) -> dict | None:
    s = world()["shipments"].get(shipment_id)
    return {"shipment_id": shipment_id, **s} if s else None
