"""
Structured output schema for the Generator.

The LLM must return JSON matching this shape exactly:
{
  "items": [
    {
      "question": "What does TCP guarantee that UDP does not?",
      "answer": "Reliable, ordered delivery of data",
      "citations": ["I2208-Part-3_p12_c1"]
    }
  ]
}

Validation happens in two layers:
1. Pydantic (shape): fields exist, types are right, lists non-empty.
2. Semantic (grounding): every citation must be a chunk_id that was
   actually in the retrieved context — checked in generator.py, because
   the schema alone can't know which chunk_ids exist.
"""
from pydantic import BaseModel, Field


class QuizItem(BaseModel):
    question: str = Field(min_length=10)
    answer: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)  # at least one chunk_id


class Quiz(BaseModel):
    items: list[QuizItem] = Field(min_length=1)
