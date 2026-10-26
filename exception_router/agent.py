"""The agent loop: the same while-loop as step1_tool_loop.py, made production-shaped.

Additions over step 1:
  * A "submit" tool gives structured output: the agent finishes by calling submit_<x>(...), and its
    input (validated by the JSON schema) *is* the result. No parsing prose.
  * If the agent stops without submitting, it gets nudged. On the last turn, submit is forced via
    tool_choice where the model supports it. Some models reject a forced tool_choice
    ("type tool/any not supported for this model"); then we fall back to tool_choice=auto plus
    the instruction and nudge, and detect the capability once per process.
  * Tool errors (unknown tool, bad arguments, exceptions) go back to the model as is_error results
    instead of crashing the run.
  * Token usage, API calls and every tool call are recorded for the eval and the trace.
"""
import json
import time
from dataclasses import dataclass, field

from .config import MAX_TOKENS, MODEL
from .tools import Tool


@dataclass
class AgentRun:
    agent: str
    output: dict | None = None
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    forced_submit: bool = False
    seconds: float = 0.0


class AgentError(RuntimeError):
    pass


_FORCING_SUPPORTED = True  # flipped to False the first time the API rejects a forced tool_choice


def _create(client, kwargs: dict):
    """Call the API. Returns (response, forced). Drops tool_choice if the model can't be forced."""
    global _FORCING_SUPPORTED
    if "tool_choice" in kwargs and not _FORCING_SUPPORTED:
        kwargs = {k: v for k, v in kwargs.items() if k != "tool_choice"}
    try:
        return client.messages.create(**kwargs), "tool_choice" in kwargs
    except Exception as e:  # noqa: BLE001
        if "tool_choice" in kwargs and getattr(e, "status_code", None) == 400 and "tool_choice" in str(e):
            _FORCING_SUPPORTED = False
            kwargs = {k: v for k, v in kwargs.items() if k != "tool_choice"}
            return client.messages.create(**kwargs), False
        raise


def _execute(tool_map: dict[str, Tool], name: str, args: dict) -> tuple[dict, bool]:
    if name not in tool_map:
        return {"error": f"Unknown tool {name}"}, True
    try:
        result = tool_map[name].fn(**args)
    except TypeError as e:  # wrong/missing arguments
        return {"error": f"Bad arguments for {name}: {e}"}, True
    except Exception as e:  # noqa: BLE001 - surface any tool failure to the model
        return {"error": f"{type(e).__name__}: {e}"}, True
    return result, isinstance(result, dict) and "error" in result


def run_agent(client, *, agent: str, system: str, tools: list[Tool], submit: dict,
              user_message: str, max_turns: int = 8, verbose: bool = False) -> AgentRun:
    run = AgentRun(agent=agent)
    t0 = time.time()
    tool_map = {t.name: t for t in tools}
    schemas = [t.schema for t in tools] + [submit]
    messages: list[dict] = [{"role": "user", "content": user_message}]
    system = (f"{system}\n\nWhen you are done, you MUST call the {submit['name']} tool with your final answer. "
              f"Never give the final answer as plain text.")

    for turn in range(max_turns):
        kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system, tools=schemas, messages=messages)
        last_turn = turn == max_turns - 1
        if last_turn or not tools:
            kwargs["tool_choice"] = {"type": "tool", "name": submit["name"]}
        resp, forced = _create(client, kwargs)
        run.forced_submit = forced and last_turn and bool(tools)
        run.api_calls += 1
        run.input_tokens += resp.usage.input_tokens
        run.output_tokens += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        done = next((b for b in tool_uses if b.name == submit["name"]), None)
        if done is not None:
            run.output = dict(done.input)
            run.seconds = round(time.time() - t0, 2)
            if verbose:
                print(f"    [{agent}] submit -> {json.dumps(run.output, ensure_ascii=False)[:300]}")
            return run

        if not tool_uses:  # the model answered in prose; ask it to submit
            messages.append({"role": "user",
                             "content": f"Do not reply in text. Call the {submit['name']} tool now with your final answer."})
            continue

        results = []
        for b in tool_uses:
            out, is_err = _execute(tool_map, b.name, dict(b.input))
            run.tool_calls.append({"tool": b.name, "input": dict(b.input), "error": is_err})
            if verbose:
                print(f"    [{agent}] {b.name}({json.dumps(dict(b.input))}){'  ERROR' if is_err else ''}")
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out, ensure_ascii=False), "is_error": is_err})
        messages.append({"role": "user", "content": results})

    raise AgentError(f"{agent}: no submission after {max_turns} turns")
