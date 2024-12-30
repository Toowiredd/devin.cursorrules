#!/usr/bin/env /workspace/tmp_windsurf/py310/bin/python3

from openai import OpenAI
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def create_llm_client(base_url="http://192.168.180.137:8006/v1", api_key="not-needed"):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    return client

def query_llm(prompt, client=None, model="Qwen/Qwen2.5-32B-Instruct-AWQ"):
    if client is None:
        client = create_llm_client()
    
    try:
        logger.info(f"Querying LLM with prompt: {prompt}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        logger.info("Successfully received response from LLM")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        logger.error("Note: If you haven't configured a local LLM server, this error is expected and can be ignored.")
        logger.error("The LLM functionality is optional and won't affect other features.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Query an LLM with a prompt')
    parser.add_argument('--prompt', type=str, help='The prompt to send to the LLM', required=True)
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ",
                       help='The model to use (default: Qwen/Qwen2.5-32B-Instruct-AWQ)')
    parser.add_argument('--base-url', type=str, default="http://192.168.180.137:8006/v1",
                       help='The base URL of the LLM server (default: http://192.168.180.137:8006/v1)')
    parser.add_argument('--api-key', type=str, default="not-needed",
                       help='The API key for the LLM server (default: not-needed)')
    args = parser.parse_args()

    client = create_llm_client(base_url=args.base_url, api_key=args.api_key)
    response = query_llm(args.prompt, client, model=args.model)
    if response:
        print(response)
    else:
        print("Failed to get response from LLM")

if __name__ == "__main__":
    main()
