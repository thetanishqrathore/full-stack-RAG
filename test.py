# network_test.py
import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Fix gRPC DNS resolution issue
os.environ['GRPC_DNS_RESOLVER'] = 'native'

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    """
    A simple test to check connectivity to Google's Generative Language API.
    """
    logging.info("--- Starting Network Connectivity Test ---")
    
    # 1. Load API Key from .env file
    try:
        logging.info("Attempting to load .env file...")
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("FAILED: GOOGLE_API_KEY not found in .env file.")
            return
        logging.info("SUCCESS: .env file loaded and API key found.")
    except Exception as e:
        logging.error(f"FAILED: Could not load .env file. Error: {e}")
        return

    # 2. Configure the Gemini client
    try:
        logging.info("Attempting to configure Google AI client...")
        genai.configure(api_key=api_key)
        logging.info("SUCCESS: Google AI client configured.")
    except Exception as e:
        logging.error(f"FAILED: Could not configure client. Error: {e}")
        return

    # 3. List available models (this makes an actual network call)
    try:
        logging.info("Attempting to make a network call to list models...")
        # This is the line that will likely fail if there's a network issue.
        models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not models:
            logging.warning("WARNING: Could not find any models, but the connection was successful.")
        else:
            # Print the first model found as proof of success
            logging.info(f"Found model: {models[0].name}")
            logging.info("--- TEST PASSED: Successfully connected to Google's API. ---")

    except Exception as e:
        logging.error("--- TEST FAILED: Could not connect to Google's API. ---")
        logging.error(f"The specific error is: {e}")

if __name__ == "__main__":
    run_test()

