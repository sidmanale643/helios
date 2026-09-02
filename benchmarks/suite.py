from dataclasses import dataclass

SUITE_VERSION = 2

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

BALANCED_CONTEXT = """
A regional library system has eight branches, a shared catalog, self-checkout kiosks, and a
mobile app. Patrons report that newly returned books sometimes remain unavailable for several
minutes, while staff occasionally see the same hold assigned twice during busy evenings. The
system uses one API, PostgreSQL, Redis, and a background worker. The team can make focused
changes but cannot replace these components. Recommend a staged reliability improvement that
includes data integrity, cache invalidation, observability, rollout safety, and user-facing error
handling. State the tradeoffs and define concrete metrics.
""".strip()

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

AGENT_USER = "Find open orders for customer ana@example.test in tenant northwind."
LOOKUP_CALL = """SIMULATED TOOL CALL
{"name":"lookup_customer","arguments":{"tenant_id":"northwind","email":"ana@example.test"}}"""
LOOKUP_RESULT = """SIMULATED TOOL RESULT: lookup_customer
{"status":"ok","customer":{"customer_id":"cus_1042","email":"ana@example.test"}}"""
ORDERS_CALL = """SIMULATED TOOL CALL
{"name":"list_orders","arguments":{"tenant_id":"northwind","customer_id":"cus_1042","status":"open","limit":10}}"""
ORDERS_RESULT = """SIMULATED TOOL RESULT: list_orders
{"status":"ok","orders":[{"order_id":"ord_731","payment_state":"paid","fulfillment_state":"processing"}],"next_cursor":null}"""
ORDER_CALL = """SIMULATED TOOL CALL
{"name":"get_order","arguments":{"tenant_id":"northwind","order_id":"ord_731"}}"""
ORDER_RESULT = """SIMULATED TOOL RESULT: get_order
{"status":"ok","order":{"order_id":"ord_731","version":4,"payment_state":"paid","fulfillment_state":"processing","line_items":[{"sku":"HEAT-24","quantity":1}]}}"""


@dataclass(frozen=True)
class Workload:
    name: str
    kind: str
    description: str
    max_new_tokens: int


@dataclass(frozen=True)
class RequestSpec:
    workload: Workload
    phase: str
    messages: tuple[tuple[str, str], ...]


WORKLOADS = (
    Workload("prefill-long", "prefill", "Long context with a short answer.", 8),
    Workload("decode-long", "decode", "Short prompt with a long continuation.", 2_048),
    Workload("balanced", "balanced", "Medium context and medium continuation.", 64),
    Workload(
        "agent-prefix",
        "prefix",
        "Growing simulated tool-agent transcript measured in order.",
        48,
    ),
)


def requests_for(workload: Workload) -> tuple[RequestSpec, ...]:
    if workload.kind == "prefill":
        context = "\n\n".join(PREFILL_CONTEXT for _ in range(5))
        return (
            RequestSpec(
                workload,
                "uncached",
                (("user", f"{context}\n\nName the first action only."),),
            ),
        )
    if workload.kind == "decode":
        return (
            RequestSpec(
                workload,
                "uncached",
                (
                    (
                        "user",
                        (
                            "Write a 2,000-word essay about how cities can prepare for extreme "
                            "heat. Continue with concrete examples until the output limit."
                        ),
                    ),
                ),
            ),
        )
    if workload.kind == "balanced":
        return (RequestSpec(workload, "uncached", (("user", BALANCED_CONTEXT),)),)
    messages: tuple[tuple[str, str], ...] = (
        ("system", AGENT_SYSTEM),
        ("user", AGENT_USER),
    )
    steps = [("cold", messages)]
    for phase, appended in (
        ("after-customer", (("assistant", LOOKUP_CALL), ("tool", LOOKUP_RESULT))),
        ("after-orders", (("assistant", ORDERS_CALL), ("tool", ORDERS_RESULT))),
        ("after-order", (("assistant", ORDER_CALL), ("tool", ORDER_RESULT))),
    ):
        messages += appended
        steps.append((phase, messages))
    return tuple(
        RequestSpec(workload, phase, transcript) for phase, transcript in steps
    )
