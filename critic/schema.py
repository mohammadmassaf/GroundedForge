"""
Structured output schema for the Critic.

For one claim (a quiz item's question+answer) checked against its cited
chunks, the Critic must return:
{
  "supported": true,
  "reason": "The chunk on p.53 states the propagation delay formula verbatim."
}

`supported` is a strict bool — "partially supported" collapses to false:
if any part of the claim isn't in the cited text, the claim is not grounded.
"""
from pydantic import BaseModel, Field


class Verdict(BaseModel):
    supported: bool
    reason: str = Field(min_length=5)
