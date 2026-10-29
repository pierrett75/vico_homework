import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from tool import tools, available_functions

load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def run_agent(user_prompt: str, model: str = "gpt-4o") -> str:
    """
    Executes a multi-turn tool calling loop until the model
    produces a final response.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant capable of analyzing text, counting tokens, and parsing JSON data.",
        },
        {"role": "user", "content": user_prompt},
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # If the model didn't request any tool calls, return its text answer
        if not response_message.tool_calls:
            return response_message.content or ""

        # Execute each requested tool call
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)

            if function_to_call:
                function_args = json.loads(tool_call.function.arguments)
                tool_output = function_to_call(**function_args)
            else:
                tool_output = {"error": f"Function {function_name} not found"}

            # Send the result back to the model as a 'tool' message
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(tool_output),
                }
            )


if __name__ == "__main__":
    test_prompt = (
        'Can you estimate the token count and cost for running this on gpt-4o: '
        '"Artificial intelligence is transforming software engineering across the globe."? '
        'Also check how many items are in this JSON: \'{"users": ["alice", "bob", "charlie"]}\' under "users".'
    )

    print("Running query...")
    answer = run_agent(test_prompt)
    print("\nAgent Output:\n")
    print(answer)