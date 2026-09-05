from dataclasses import dataclass

from helios.runtime.warmup import COMPILE_WARMUP_PROMPT

SUITE_VERSION = 4

PREFILL_CONTEXT = """
Northwind Health operates twelve neighborhood clinics. Each clinic publishes appointment
availability, accepts insurance documents, sends reminders, and coordinates referrals with
external specialists. The booking service uses PostgreSQL as its source of truth, Redis for
short-lived availability reads, and a worker queue for reminders and document processing.
Traffic is normally steady but rises sharply on Monday mornings and after severe weather.
Recent incidents include duplicate reservations during retries, stale availability after bulk
schedule imports, slow searches when one clinic adds thousands of slots, and reminder failures
that were reported to patients as booking failures. The team has six engineers, no dedicated
site-reliability group, and a requirement to preserve the current public API. Deployments happen
twice a week. Logs include request IDs but do not consistently connect HTTP requests, database
transactions, queue jobs, and provider calls. The team can add indexes, constraints, metrics,
traces, retry policies, and small background jobs, but cannot replace the database or split the
service into microservices this quarter. Product leadership wants fewer booking errors without
slowing down the normal appointment flow. Security requires uploaded insurance documents to
remain encrypted and access-controlled. Support needs error messages that distinguish a saved
booking from a failed notification. The plan must include rollout order, measurable success
criteria, rollback signals, ownership, and a practical way to test traffic spikes before release.
""".strip()

PREFILL_INPUT = "\n\n".join(PREFILL_CONTEXT for _ in range(5)) + (
    "\n\nName the first action only."
)

BALANCED_INPUT = COMPILE_WARMUP_PROMPT

BALANCED_INPUTS = (
    "Explain how to roll out a database index safely, including how to measure whether it helped.",
    "Draft a concise incident update for a partial API outage, including current impact and the next checkpoint.",
    "Compare optimistic locking and pessimistic locking for a booking system with concurrent updates.",
    "Outline a practical plan to find and reduce duplicate background jobs in a Python service.",
    "Explain how a service can distinguish an empty search result from an unavailable dependency.",
    "Propose three metrics for detecting a growing queue backlog and explain what each one reveals.",
    "Describe a safe migration path from a single cache key to versioned cache keys.",
    "Write a short runbook for responding to elevated database connection-pool exhaustion.",
    "Explain how idempotency keys protect payment or refund operations during client retries.",
    "Suggest a staged approach to adding tracing to an API and its background workers.",
)

AGENT_SYSTEM = """
You are an operations assistant for a multi-tenant commerce platform. Decide which tools to use,
keep tenant data isolated, and explain the final action briefly. Never invent tool results. Read
operations are safe. Any mutation requires a confirmed tenant identifier and an explicit user
request. If a dependency is unavailable, report the unavailable dependency separately from an
empty result. Preserve idempotency keys across retries and do not retry validation errors.

Available tools:
- lookup_customer: accepts tenant_id and one of customer_id or email; returns a customer record.
- list_orders: accepts tenant_id, customer_id, status, created_after, cursor, and limit; returns
  matching orders plus a pagination cursor.
- get_order: accepts tenant_id and order_id; returns line items, payment state, fulfillment state,
  version, and prior adjustments.
- search_inventory: accepts tenant_id, sku, warehouse_region, and minimum_quantity; returns
  inventory candidates with observation timestamps.
- quote_refund: accepts tenant_id, order_id, line_item_ids, reason, and shipping_refund policy;
  returns a non-binding refund calculation and warnings.
- create_refund: accepts tenant_id, order_id, quote_id, expected_order_version, idempotency_key,
  and confirmation; returns the committed adjustment or a version conflict.
- cancel_fulfillment: accepts tenant_id, order_id, fulfillment_id, expected_state, idempotency_key,
  and confirmation; returns the updated fulfillment state.
- create_support_note: accepts tenant_id, customer_id, order_id, category, summary, and visibility;
  returns a note identifier.

For each request, first identify missing inputs. Then produce a compact plan containing only the
necessary tool calls. For mutations, read the current order immediately before writing, pass its
version to the mutation, and stop on conflicts instead of guessing. Do not expose internal IDs or
private notes belonging to another tenant. A tool returning no records is a valid empty result;
a timeout, authentication error, rate limit, or malformed response is a provider failure.
""".strip()

AGENT_INPUT = "Find open orders for customer ana@example.test in tenant northwind."
LOOKUP_CALL = """
{"name":"lookup_customer","arguments":{"tenant_id":"northwind","email":"ana@example.test"}}"""
LOOKUP_RESULT = """
{"status":"ok","customer":{"customer_id":"cus_1042","email":"ana@example.test"}}"""
ORDERS_CALL = """
{"name":"list_orders","arguments":{"tenant_id":"northwind","customer_id":"cus_1042","status":"open","limit":10}}"""
ORDERS_RESULT = """
{"status":"ok","orders":[{"order_id":"ord_731","payment_state":"paid","fulfillment_state":"processing"}],"next_cursor":null}"""
ORDER_CALL = """
{"name":"get_order","arguments":{"tenant_id":"northwind","order_id":"ord_731"}}"""
ORDER_RESULT = """
{"status":"ok","order":{"order_id":"ord_731","version":4,"payment_state":"paid","fulfillment_state":"processing","line_items":[{"sku":"HEAT-24","quantity":1}]}}"""

Message = tuple[str, str]


@dataclass(frozen=True)
class ToolExchange:
    phase: str
    call: str
    result: str


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    input: tuple[Message, ...]
    max_new_tokens: int
    tool_exchanges: tuple[ToolExchange, ...] = ()


WORKLOADS = (
    # Workload(
    #     "prefill-long",
    #     "Long input with a short output.",
    #     (("user", PREFILL_INPUT),),
    #     8,
    # ),
    Workload(
        "decode-long",
        "Short input with a long output.",
        (("user", "Write a 2,000-word essay about how cities can prepare for extreme heat. Continue with concrete examples until the output limit."),),
        2_048,
    ),
    Workload(
        "balanced",
        "Medium input with a medium output.",
        (("user", BALANCED_INPUT),),
        64,
    ),
    *(
        Workload(
            f"balanced-{index}",
            "Medium input with a medium output.",
            (("user", prompt),),
            64,
        )
        for index, prompt in enumerate(BALANCED_INPUTS, start=1)
    ),
    Workload(
        "agent-prefix",
        "A growing transcript with simulated tool calls and results.",
        (("system", AGENT_SYSTEM), ("user", AGENT_INPUT)),
        48,
        (
            ToolExchange("after-customer", LOOKUP_CALL, LOOKUP_RESULT),
            ToolExchange("after-orders", ORDERS_CALL, ORDERS_RESULT),
            ToolExchange("after-order", ORDER_CALL, ORDER_RESULT),
        ),
    ),
)


@dataclass(frozen=True)
class RequestSpec:
    workload: Workload
    phase: str
    messages: tuple[Message, ...]


def requests_for(workload: Workload) -> tuple[RequestSpec, ...]:
    if not workload.tool_exchanges:
        return (RequestSpec(workload, "uncached", workload.input),)

    messages = workload.input
    requests = [RequestSpec(workload, "cold", messages)]
    for exchange in workload.tool_exchanges:
        messages += (("assistant", exchange.call), ("tool", exchange.result))
        requests.append(RequestSpec(workload, exchange.phase, messages))
    return tuple(requests)
