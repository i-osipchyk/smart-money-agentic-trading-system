# ADR-003: Trade Validation Agent Replacing Two-Agent Architecture

## Status
Accepted

## Context
The PoC (see docs/poc-results.md) validated that the core signal detection logic works, but revealed two problems with the two-agent trade-finding architecture:

1. The HTF agent produced ambiguous reasoning when multiple FVGs were present, often identifying both bullish and bearish POIs simultaneously.
2. This ambiguity cascaded to the LTF agent, which consistently hesitated and produced low-confidence decisions.

The root cause: asking an LLM to find a trade from raw signals is too open-ended. The output space is too large, making it hard to evaluate and hard to trust.

## Decision
Replace the two-agent architecture with a single Strategies and Trade Validation Agent.

The strategy combines signals and different timeframes, and idetifies all potential trades and returns them in StrategySetup model format. This is used by the agent.

The agent receives a setup based on strategy and defines a TradeDecision (entry, SL, TP, direction). The agent's job is to validate whether the setup aligns with current market structure and find targets based on defined risk management guidelines.

The old two-agent code is removed from the main codebase. It is preserved in git history and documented in docs/poc-results.md.

## Consequences

**Positive:**
- Agent has a single, clearly bounded responsibility
- Output is easier to evaluate (P&L vs open-ended text)
- Maps to real usage: trader spots setup, system pressure-tests it
- Simpler LangGraph graph — one node, no conditional edges

**Negative:**
- System no longer autonomously finds setups — requires definition with the code
- Reduces automation potential for future live trading phase
