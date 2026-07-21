import os
import json
from datetime import datetime
from dateutil import parser

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "array[string]",
    "array[integer]"
}


class DynamicRequest(BaseModel):
    text: str
    schema: dict


def convert_value(value, typ):
    if value is None:
        return None

    try:

        if typ == "string":
            return str(value)

        if typ == "integer":
            return int(value)

        if typ == "float":
            return float(value)

        if typ == "boolean":
            if isinstance(value, bool):
                return value

            if str(value).lower() in ["true", "yes", "1"]:
                return True

            if str(value).lower() in ["false", "no", "0"]:
                return False

            return None

        if typ == "date":
            d = parser.parse(str(value))
            return d.strftime("%Y-%m-%d")

        if typ == "array[string]":

            if isinstance(value, list):
                return [str(x) for x in value]

            return [str(value)]

        if typ == "array[integer]":

            if isinstance(value, list):
                return [int(x) for x in value]

            return [int(value)]

    except Exception:
        return None

    return None


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    schema = req.schema

    prompt = f"""
You are an information extraction system.

Extract ONLY the fields listed in the schema.

Return ONLY valid JSON.

Rules:

- No markdown
- No explanation
- No extra keys
- Missing values -> null
- Arrays should be JSON arrays
- Dates should be ISO if possible

Schema:

{json.dumps(schema, indent=2)}

Text:

{req.text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        content = content.replace("json", "").strip()

    try:
        extracted = json.loads(content)
    except:
        extracted = {}

    result = {}

    for key, typ in schema.items():

        if typ not in SUPPORTED_TYPES:
            result[key] = None
            continue

        result[key] = convert_value(
            extracted.get(key),
            typ
        )

    return result


@app.get("/")
def root():
    return {"status": "running"}