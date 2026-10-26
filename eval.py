"""Evaluate the router on the 20 labeled tickets in data/tickets.json.

  python eval.py                    # normal routing (rule decides)
  python eval.py --force single     # ablation: everything through the single resolver
  python eval.py --force multi      # ablation: everything through the 3-specialist pipeline
  python eval.py --only T02,T18     # subset

Metrics (per ticket, then averaged):
  types      triage got exactly the expected set of exception types
  route      the routing rule chose the expected route (skipped when --force is used)
  root_cause fraction of expected exception types whose root cause is correct
  escalate   escalation decision matches the policy-derived label
  consistent the escalation decision matches what was actually filed (decision vs action)
  all_correct types + root causes + escalation all right
Cost: API calls, tool calls, tokens and seconds per ticket, split by route.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from exception_router.config import RESULTS_DIR, make_client
from exception_router.data import load_tickets
from exception_router.orchestrator import handle_ticket


def score(ticket: dict, res: dict, forced: bool) -> dict:
    exp = ticket["expected"]
    exp_rc = exp["root_causes"]
    got_rc = res["root_causes"]
    rc_hits = sum(got_rc.get(t) == rc for t, rc in exp_rc.items())
    s = {
        "types": set(res["triage"]["exception_types"]) == set(exp["exception_types"]),
        "route": None if forced else res["rule_route"] == exp["route"],
        "root_cause": rc_hits / len(exp_rc),
        "escalate": bool(res["decision"]["escalate"]) == exp["escalate"],
        "consistent": res["escalation_consistent"],
    }
    s["all_correct"] = s["types"] and s["root_cause"] == 1.0 and s["escalate"]
    return s


def run_one(client, t, force):
    try:
        res = handle_ticket(client, t, force_route=force)
        return {"ticket_id": t["id"], "expected": t["expected"], "result": res,
                "score": score(t, res, bool(force))}
    except Exception as e:  # noqa: BLE001 - one failure shouldn't kill the eval
        return {"ticket_id": t["id"], "expected": t["expected"], "error": f"{type(e).__name__}: {e}"}


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", choices=["single", "multi"])
    ap.add_argument("--only", help="Comma-separated ticket IDs")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tickets = load_tickets()
    if args.only:
        keep = set(args.only.split(","))
        tickets = [t for t in tickets if t["id"] in keep]

    client = make_client()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda t: run_one(client, t, args.force), tickets))

    ok = [r for r in rows if "score" in r]
    errors = [r for r in rows if "error" in r]
    label = args.force or "routed"

    print(f"\n{'ID':4} {'route':6} {'types':5} {'rc':4} {'esc':4} {'cons':4}  calls  tokens   notes")
    for r in rows:
        if "error" in r:
            print(f"{r['ticket_id']:4} ERROR  {r['error']}")
            continue
        s, res, st = r["score"], r["result"], r["result"]["stats"]
        notes = []
        if not s["types"]:
            notes.append(f"types={res['triage']['exception_types']}")
        if s["route"] is False:
            notes.append(f"route={res['rule_route']} expected {r['expected']['route']}")
        if s["root_cause"] < 1:
            notes.append(f"rc={res['root_causes']} expected {r['expected']['root_causes']}")
        if not s["escalate"]:
            notes.append(f"escalate={res['decision']['escalate']} expected {r['expected']['escalate']}")
        if not s["consistent"]:
            notes.append(f"decision/filed mismatch (filed={res['escalations_filed']})")
        yn = lambda v: " -  " if v is None else ("ok  " if v is True or v == 1.0 else "XX  ")
        print(f"{r['ticket_id']:4} {res['route']:6} {yn(s['types'])} {yn(s['root_cause'])}{yn(s['escalate'])}"
              f"{yn(s['consistent'])}  {st['api_calls']:5}  {st['input_tokens'] + st['output_tokens']:6}   "
              f"{'; '.join(notes)}")

    summary = {
        "label": label,
        "n": len(rows), "errors": len(errors),
        "types": pct([r["score"]["types"] for r in ok]),
        "route": pct([r["score"]["route"] for r in ok]),
        "root_cause": pct([r["score"]["root_cause"] for r in ok]),
        "escalate": pct([r["score"]["escalate"] for r in ok]),
        "consistent": pct([r["score"]["consistent"] for r in ok]),
        "all_correct": pct([r["score"]["all_correct"] for r in ok]),
        "by_route": {},
    }
    for route in ("single", "multi"):
        rr = [r for r in ok if r["result"]["route"] == route]
        if rr:
            n = len(rr)
            summary["by_route"][route] = {
                "tickets": n,
                "all_correct": pct([r["score"]["all_correct"] for r in rr]),
                "avg_api_calls": round(sum(r["result"]["stats"]["api_calls"] for r in rr) / n, 1),
                "avg_tokens": round(sum(r["result"]["stats"]["input_tokens"] + r["result"]["stats"]["output_tokens"]
                                        for r in rr) / n),
                "avg_seconds": round(sum(r["result"]["stats"]["seconds"] for r in rr) / n, 1),
            }

    print(f"\n=== SUMMARY ({label}, n={summary['n']}, errors={summary['errors']})")
    for k in ("types", "route", "root_cause", "escalate", "consistent", "all_correct"):
        print(f"  {k:12} {summary[k]}")
    for route, v in summary["by_route"].items():
        print(f"  [{route}] {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{label}.json"
    out.write_text(json.dumps({"run_at": datetime.now().isoformat(timespec="seconds"),
                               "summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
