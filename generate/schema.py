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

    def all_citations(self) -> list[str]:
      out = []
      for item in self.items:
          for cid in item.citations:
              out.append(cid)

      return out

class CVBullet(BaseModel):
    """One resume bullet: a first-person claim about work you actually did."""
    text: str = Field(min_length=10)
    citations: list[str] = Field(min_length=1)   # at least one chunk_id


class Bullets(BaseModel):
    bullets: list[CVBullet] = Field(default_factory = list)
    def all_citations(self) -> list[str]:
        out = []
        for bullet in self.bullets:
            for cid in bullet.citations:
                out.append(cid)
        return out
    def is_empty(self)-> bool:
        return not self.bullets


class StarSection(BaseModel):
    """One part of a STAR answer, citing only its own evidence pool."""
    text: str = Field(min_length=10)
    citations: list[str] = Field(min_length=1)


class STARAnswer(BaseModel):
    question: str = Field(min_length=5)
    situation: StarSection
    task: StarSection
    action: StarSection
    result: StarSection

    def sections(self) -> list[tuple[str, StarSection]]:
        return [("Situation", self.situation), ("Task", self.task),
                ("Action", self.action), ("Result", self.result)]

    def all_citations(self) -> list[str]:
        out = []
        for _, section in self.sections():
            for cid in section.citations:
                out.append(cid)
        return out


class GuideClaim(BaseModel):
    text : str = Field(min_length=10)
    citations : list[str] = Field(min_length=1)
class GuideSection(BaseModel):
    heading : str = Field(min_length=1)
    claims : list[GuideClaim] = Field(min_length=1)
class Guide(BaseModel):
    sections : list[GuideSection] = Field(min_length=1)

    def all_citations(self) -> list[str]:
        out =[]
        for section in self.sections:
            for claim in section.claims:
                for cid in claim.citations:
                    out.append(cid)
        return out 