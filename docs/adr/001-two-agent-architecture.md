# ADR-001: Two-Agent Architecture (HTF + LTF)

## Status
Accepted

## Context
A Smart Money trading system needs to handle two distinct concerns:

1. **Macro analysis** — identifying market structure, trend direction, and Points of Interest (POI). This operates on higher timeframes (4H, 1D), changes slowly, and requires broad contextual reasoning.
2. **Trade execution decisions** — finding confirmation of a setup and defining entry, SL, and TP. This operates on lower timeframes (15m, 5m), reacts quickly, and requires precision over breadth.

Combining both in a single agent creates conflicting update frequencies, mixed responsibilities, and makes the system harder to reason about, test, and improve independently.

## Decision
Split responsibilities into two agents connected by a shared `MarketState` object:

- **HTF Agent** — consumes higher timeframe candles, detects market structure (BOS, FVG, fractals), classifies trend, and identifies POIs. Writes its conclusions to `MarketState`.
- **LTF Agent** — reads `MarketState` to understand the macro context, then consumes lower timeframe candles and looks for confirmation. Produces a `TradeDecision` (enter, skip, or no setup).

Agents communicate exclusively through `MarketState` — never directly. LangGraph orchestrates the execution order: HTF always runs first, LTF only runs if HTF found a valid POI.

## Consequences

**Positive:**
- Each agent has a single, clearly defined responsibility
- HTF logic can be developed and tested independently of LTF
- Adding a third agent (e.g. risk management) later requires no changes to existing agents
- Failure in one agent is isolated and easy to diagnose

**Negative:**
- `MarketState` becomes a critical shared dependency — its schema must be stable
- Adds coordination overhead: LTF must handle the case where HTF produced no POI
- Two agents means two sets of prompts to maintain and tune
