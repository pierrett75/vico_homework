import json
from typing import Any, Dict, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None


def count_tokens(text: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Count the number of tokens in a text string for a given OpenAI model,
    along with an estimated input cost in USD.
    """
    pricing_per_1m = {
        "gpt-4o": 2.50,
        "gpt-4o-mini": 0.15,
        "gpt-4.1": 2.00,
        "gpt-4.1-mini": 0.40,
        "o3-mini": 1.10,
    }

    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        token_count = len(encoding.encode(text))
        method = "tiktoken"
    else:
        token_count = max(1, len(text) // 4) if text else 0
        method = "approximate (install tiktoken for exact count)"

    price_per_1m = pricing_per_1m.get(model, 2.50)
    estimated_cost_usd = round((token_count / 1_000_000) * price_per_1m, 8)

    return {
        "model": model,
        "token_count": token_count,
        "character_count": len(text),
        "estimated_input_cost_usd": estimated_cost_usd,
        "counting_method": method,
    }


def count_json_items(json_string: str, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a JSON string and count the number of items/records inside it
    to help estimate batch processing size and cost.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}", "item_count": 0}

    target = data
    if key:
        if isinstance(data, dict) and key in data:
            target = data[key]
        else:
            return {
                "error": f"Key '{key}' not found in root JSON object.",
                "item_count": 0,
            }

    if isinstance(target, list):
        count = len(target)
        target_type = "list"
    elif isinstance(target, dict):
        count = len(target.keys())
        target_type = "dict_keys"
    else:
        count = 1
        target_type = type(target).__name__

    return {
        "key_inspected": key or "<root>",
        "target_type": target_type,
        "item_count": count,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "count_tokens",
            "description": "Use this function to count the number of tokens in a text string for a given OpenAI model and estimate its input cost in USD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The prompt or document text to count tokens for.",
                    },
                    "model": {
                        "type": "string",
                        "description": "The OpenAI model name, e.g. 'gpt-4o' or 'gpt-4o-mini'.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_json_items",
            "description": "Use this function to parse a JSON string and count how many items or keys it contains for batch budgeting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {
                        "type": "string",
                        "description": "A valid JSON string to inspect.",
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional dictionary key to count items from if the JSON root is an object.",
                    },
                },
                "required": ["json_string"],
            },
        },
    },
]

available_functions = {
    "count_tokens": count_tokens,
    "count_json_items": count_json_items,
}
