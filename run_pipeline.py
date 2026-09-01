"""Answer customer questions using a local knowledge base and an LLM."""

import argparse, json
import re
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
KNOWLEDGE_FILE = Path("knowledge_base.json")
OUTPUT_FILE = Path("latest_answer.json")
def validate_input(user_query: str) -> str:
    cleaned = " ".join(user_query.split())
    if not cleaned:
        raise ValueError("Question cannot be empty.")
    if len(cleaned) > 500:
        raise ValueError("Question must be 500 characters or fewer.")
    return cleaned

def retrieve_context(query: str, limit: int = 3) -> list[dict]:
    articles = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    query_words = set(re.findall(r"\w+", query.lower()))
    def relevance(article: dict) -> int:
        words = set(re.findall(r"\w+", article["text"].lower()))
        return len(query_words & words)
    ranked = sorted(articles, key=relevance, reverse=True)
    return [article for article in ranked[:limit] if relevance(article) > 0]

def build_prompt(query: str, context: list[dict]) -> str:
    sources = "\n".join(
        f'[{item["id"]}] {item["title"]}: {item["text"]}' for item in context
    ) or "No relevant knowledge-base articles were found."
    return f"""Customer question: {query}
Knowledge base: {sources}
Return only JSON with these fields:
{{"answer": "helpful answer", "source_ids": ["used article IDs"],
  "needs_human": false}}
Use only the knowledge base. Set needs_human to true when it lacks the answer."""

def call_llm(prompt: str) -> str:
    client = OpenAI()
    response = client.responses.create(
        model="gpt-5-mini",
        instructions="You are a concise and accurate customer-support assistant.",
        input=prompt,
    )
    return response.output_text

def parse_response(response: str) -> dict:
    return json.loads(response)

def validate_output(result: dict, valid_ids: set[str]) -> dict:
    required = {"answer", "source_ids", "needs_human"}
    if not required.issubset(result):
        raise ValueError(f"Model output is missing: {required - result.keys()}")
    if not isinstance(result["answer"], str) or not result["answer"].strip():
        raise ValueError("Model returned an empty answer.")
    if not set(result["source_ids"]).issubset(valid_ids):
        raise ValueError("Model cited an unknown knowledge-base article.")
    if not isinstance(result["needs_human"], bool):
        raise ValueError("needs_human must be true or false.")
    return result

def run_pipeline(user_query: str) -> dict:
    cleaned = validate_input(user_query)
    context = retrieve_context(cleaned)
    response = call_llm(build_prompt(cleaned, context))
    result = validate_output(parse_response(response), {x["id"] for x in context})
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="customer question to answer")
    print(json.dumps(run_pipeline(parser.parse_args().question), indent=2))
