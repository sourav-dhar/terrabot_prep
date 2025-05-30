LangGraph Canvas Structure
State Schema
pythonclass State(TypedDict):
    # Input
    user_query: str
    
    # Query Analysis
    query_type: Optional[str]  # status_check, cancellation, reversal, policy_question, tat_inquiry
    transaction_ids: List[str]
    
    # Agent Results
    api_responses: List[Dict[str, Any]]
    rag_contexts: List[Dict[str, Any]]
    
    # Response
    draft_response: Optional[str]
    final_response: Optional[str]
    confidence_score: float
    
    # Control Flow
    next_agent: Optional[str]
    error_count: int
    requires_human: bool
    
    # Messages
    messages: List[BaseMessage]
Nodes (7 Total)
1. supervisor

Purpose: Analyzes query and routes to appropriate agents
Inputs: user_query
Outputs: query_type, transaction_ids, next_agent
Next: Routes to api_agent, rag_agent, or response_builder

2. api_agent

Purpose: Fetches transaction data via API
Inputs: transaction_ids
Outputs: api_responses
Next: Routes to rag_agent or response_builder

3. rag_agent

Purpose: Searches policy documents
Inputs: user_query, query_type, api_responses (for context)
Outputs: rag_contexts
Next: Always goes to response_builder

4. response_builder

Purpose: Synthesizes final response from all sources
Inputs: api_responses, rag_contexts, query_type
Outputs: draft_response, confidence_score
Next: Routes to validator or END

5. validator

Purpose: Validates response against business rules
Inputs: draft_response, query_type, api_responses
Outputs: final_response, requires_human
Next: Routes to human_review or END

6. human_review

Purpose: Handles cases requiring human intervention
Inputs: draft_response, requires_human reason
Outputs: final_response
Next: Always goes to END

7. error_handler

Purpose: Handles errors and retries
Inputs: Error context from any node
Outputs: Recovery strategy
Next: Routes back to failed node or human_review

Entry Point

START → supervisor

Edges
Direct Edges (Fixed Paths)

rag_agent → response_builder
human_review → END

Conditional Edges
From supervisor:
pythondef supervisor_router(state):
    if state["query_type"] == "policy_question":
        return "rag_agent"
    elif state["transaction_ids"]:
        return "api_agent"
    else:
        return "rag_agent"  # Default for ambiguous queries

→ api_agent (if transaction IDs found)
→ rag_agent (if pure policy question)
→ error_handler (if error)

From api_agent:
pythondef api_agent_router(state):
    query_type = state["query_type"]
    if query_type in ["cancellation", "reversal", "tat_inquiry"]:
        return "rag_agent"  # Need policy context
    elif state["api_responses"]:
        return "response_builder"
    else:
        return "error_handler"

→ rag_agent (if cancellation/reversal/TAT - needs policy)
→ response_builder (if simple status check)
→ error_handler (if API fails)

From response_builder:
pythondef response_builder_router(state):
    if state["confidence_score"] < 0.8:
        return "validator"
    elif state["query_type"] in ["cancellation", "reversal"]:
        return "validator"  # Always validate critical operations
    else:
        return "END"

→ validator (if low confidence or critical operation)
→ END (if high confidence simple query)

From validator:
pythondef validator_router(state):
    if state["requires_human"]:
        return "human_review"
    else:
        return "END"

→ human_review (if validation fails or critical issue)
→ END (if validation passes)

From error_handler:
pythondef error_handler_router(state):
    if state["error_count"] > 3:
        return "human_review"
    elif state.get("last_error_node") == "api_agent":
        return "rag_agent"  # Try alternative path
    else:
        return "response_builder"  # Try to build with what we have

→ human_review (if too many errors)
→ rag_agent (as fallback if API fails)
→ Original node (retry) or alternative path

Interrupt Points

Before: human_review (allows manual intervention)

Common Query Flows
1. Simple Status Check
START → supervisor → api_agent → response_builder → END
2. Cancellation Request
START → supervisor → api_agent → rag_agent → response_builder → validator → END/human_review
3. Policy Question
START → supervisor → rag_agent → response_builder → END
4. TAT Inquiry
START → supervisor → api_agent → rag_agent → response_builder → validator → END
5. Error Recovery Flow
START → supervisor → api_agent (fails) → error_handler → rag_agent → response_builder → END
Canvas Configuration Tips

Node Colors:

supervisor: Blue (orchestrator)
api_agent, rag_agent: Orange (data fetchers)
response_builder: Green (synthesizer)
validator: Pink (quality control)
human_review: Red (escalation)
error_handler: Dark Orange (recovery)


Edge Types:

Solid lines: Normal flow
Dashed lines: Error/fallback paths
Red lines: Human escalation


Node Positions:

Place supervisor at top center
api_agent and rag_agent in parallel below supervisor
response_builder centered below them
validator to the right of response_builder
human_review at bottom right
error_handler on the side



Minimal Implementation Example
pythonfrom langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict, Any

class State(TypedDict):
    user_query: str
    query_type: Optional[str]
    transaction_ids: List[str]
    api_responses: List[Dict[str, Any]]
    rag_contexts: List[Dict[str, Any]]
    draft_response: Optional[str]
    final_response: Optional[str]
    confidence_score: float
    next_agent: Optional[str]
    error_count: int
    requires_human: bool

# Initialize graph
workflow = StateGraph(State)

# Add nodes (implement these functions)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("api_agent", api_agent_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("response_builder", response_builder_node)
workflow.add_node("validator", validator_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("error_handler", error_handler_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Add conditional edges
workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "api_agent": "api_agent",
        "rag_agent": "rag_agent",
        "error_handler": "error_handler"
    }
)

workflow.add_conditional_edges(
    "api_agent",
    api_agent_router,
    {
        "rag_agent": "rag_agent",
        "response_builder": "response_builder",
        "error_handler": "error_handler"
    }
)

# Add direct edges
workflow.add_edge("rag_agent", "response_builder")
workflow.add_edge("human_review", END)

# Compile
app = workflow.compile()
