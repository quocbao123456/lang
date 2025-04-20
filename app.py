from flask import Flask, request, jsonify
import os
import json
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
from pydantic import BaseModel, Field
from typing import Union
from datetime import datetime
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory=ConversationBufferMemory(ai_prefix="AI Assistant"),

def clean_output(output):
    """
    Extracts the JSON content from a string even if it contains markdown formatting.
    It uses a regex to find the first "{" and the last "}" and returns only that part.
    """
    # Use regex to locate the JSON object in the output.
    match = re.search(r'({.*})', output, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str.strip()
    # If no JSON object is found, return the original output stripped.
    return output.strip()
def convert_iso_to_unix(iso_string):
    """
    Converts an ISO 8601 string (e.g., '2025-04-17T00:00:00Z') to a Unix timestamp in milliseconds.
    Assumes the input is in UTC.
    """
    try:
        # Parse the ISO 8601 string
        dt = datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%SZ")
        unix_timestamp = int(dt.timestamp() * 1000)  # Convert to milliseconds
        return unix_timestamp
    except ValueError as e:
        print(f"Error parsing ISO string: {e}")
        return None

app = Flask(__name__)

# Define the expected response schema with a ResponseSchema list.
response_schemas = [
    ResponseSchema(
        name="className",
        description="The name of the class that handles the request, e.g., LessonCalendarHandler"
    ),
    ResponseSchema(
        name="function", 
        description="The name of the function to call, e.g., getCountLessonsCalendarDeserializeV2"
    ),
    ResponseSchema(
        name="params", 
        description=("A JSON object of parameters specific to the handler."
        "This structure is dynamic and may vary depending on the user request.")
    ),
]

today = datetime.utcnow().strftime("%Y-%m-%d")

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

with open('prompt_template.txt', 'r') as file:
    prompt_template_text = file.read()

PROMPT = PromptTemplate(input_variables=["history", "input"], template=prompt_template_text)
# # Build the chain using the prompt template and the LLM.
conv = ConversationChain(
    llm=OpenAI(
        temperature=0,
    ),
    prompt=PROMPT,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

@app.route('/parseIntent', methods=['POST'])
def parse_intent():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    output = ""
    try:
        # Run the LangChain with both the user input and format instructions.
        try:
            output = conv.predict(input=user_message)
            # output = conv.run(user_input=user_message,format_instructions=format_instructions,today=today)
        except Exception as e:
            print("An error occurred:", e)

        # Check if the output is a string or a dict
        if isinstance(output, str):
            cleaned_output = clean_output(output)
            return cleaned_output, 200

        elif isinstance(output, dict):
            cleaned_output = output  # No need to clean if it's already a dict
        else:
            return jsonify({"error": "Unexpected output type from chain.run"}), 500
        
        # Attempt to load the output as JSON.
        try:
            if isinstance(cleaned_output, str):
                json_output = json.loads(cleaned_output)
            else:
                json_output = cleaned_output
                
            print("Resjson_outputponse: ", json_output)

            # Convert ISO 8601 dates to Unix timestamps
            if "params" in json_output:
                if "fromDate" in json_output["params"]:
                    iso_date = json_output["params"]["fromDate"]
                    unix_timestamp = convert_iso_to_unix(iso_date)
                    if unix_timestamp is not None:
                        json_output["params"]["fromDate"] = unix_timestamp

                if "toDate" in json_output["params"]:
                    iso_date = json_output["params"]["toDate"]
                    unix_timestamp = convert_iso_to_unix(iso_date)
                    if unix_timestamp is not None:
                        json_output["params"]["toDate"] = unix_timestamp
        except json.JSONDecodeError:
            # Return an error message including the raw output for debugging.
            return jsonify({
                "error": "Failed to parse output as JSON",
                "raw_output": output
            }), 500

        return jsonify(json_output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app. In production, use a proper WSGI server and HTTPS.
    app.run(host='0.0.0.0', port=5000, debug=True)
