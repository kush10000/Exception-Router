"""
step1_tool_loop.py — the bare tool-calling loop. Nothing else.

Goal: SEE the protocol with your own eyes.
  model returns tool_use  ->  YOUR code runs the function  ->  you send tool_result  ->  repeat until end_turn

Run:  python step1_tool_loop.py
"""
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()  # copies KEY=value lines from .env into os.environ. The SDK only reads os.environ.
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"  # or whatever model rag.py uses

# --- mock data: pretend this is the client's TMS / shipment tracker ---
SHIPMENTS = {
    "SH-1001": {"status": "in_transit", "promised_date": "2026-10-05",
                "carrier": "BlueDart", "last_scan": "Nagpur hub, 2026-10-04"},
    "SH-1002": {"status": "delivered", "promised_date": "2026-10-07",
                "carrier": "Delhivery", "last_scan": "Delivered, 2026-10-07"},
}


def get_shipment(shipment_id: str) -> dict:
    return SHIPMENTS.get(shipment_id, {"error": f"no shipment {shipment_id}"})


# 1) The tool SCHEMA. This JSON is ALL the model ever sees of your function.
#    The description is a prompt: it decides WHEN the model calls the tool.
TOOLS = [{
    "name": "get_shipment",
    "description": ("Look up a shipment's current status, promised delivery date, "
                    "carrier and last scan. Use when a ticket mentions a shipment ID."),
    "input_schema": {
        "type": "object",
        "properties": {"shipment_id": {"type": "string", "description": "e.g. SH-1001"}},
        "required": ["shipment_id"],
    },
}]

# 2) Dispatch table: tool name -> real Python function.
#    The model never executes anything. It only ASKS. Your code decides whether to run it.
TOOL_FNS = {"get_shipment": get_shipment}


def run(user_msg: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": user_msg}]

    for turn in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
        )
        print(f"\n--- API call {turn + 1} | stop_reason={resp.stop_reason}")

        # 3) Append the assistant's FULL content (text + tool_use blocks) to history.
        #    The API is stateless: the conversation lives in this list, not on the server.
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            final = "".join(b.text for b in resp.content if b.type == "text")
            print("FINAL:", final)
            return final

        # 4) Execute every tool_use block and collect results.
        tool_results = []
        for block in resp.content:
            if block.type == "text":
                print("model says:", block.text)
            elif block.type == "tool_use":
                print(f"model asks: {block.name}({block.input})   id={block.id}")
                result = TOOL_FNS[block.name](**block.input)
                print("  we return:", result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # must match, or the API rejects the request
                    "content": json.dumps(result),
                })

        # 5) Tool results go back as a USER message. Then loop.
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("hit max_turns without end_turn")


if __name__ == "__main__":
    run("Ticket #42: customer says SH-1001 hasn't arrived. Is it late? "
        "Check SH-1002 from the same order too.")
