import logging

import requests
from flask import Flask, abort, jsonify, request

app = Flask(__name__)

# Configure logging to output to a file
logging.basicConfig(
    filename="flask_server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@app.route("/", methods=["POST"])
def proxy_request():
    # Get the URL, headers and data from the request
    target_url = request.json.get("url")
    headers = request.json.get("headers", {})  # Default to empty dict if headers not provided
    data = request.json.get("data")  # This will be None if not provided

    # Log the received request data
    logging.info(f"Received request: {request.json}")

    try:
        # If data is provided, use POST; otherwise use GET
        if data is not None:
            logging.info(f"Making POST request to {target_url}")
            with requests.post(target_url, headers=headers, json=data, timeout=10) as response:
                response.raise_for_status()
                return response.content
        else:
            logging.info(f"Making GET request to {target_url}")
            with requests.get(target_url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                return response.content
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return (
            jsonify(
                {
                    "error": "HTTP error occurred",
                    "message": str(http_err),
                    "status_code": response.status_code,
                    "received_request": request.json,
                }
            ),
            response.status_code,
        )
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred: {conn_err}")
        return (
            jsonify(
                {
                    "error": "Connection error occurred",
                    "message": str(conn_err),
                    "received_request": request.json,
                }
            ),
            500,
        )
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred: {timeout_err}")
        return (
            jsonify(
                {
                    "error": "Timeout error occurred",
                    "message": str(timeout_err),
                    "received_request": request.json,
                }
            ),
            500,
        )
    except requests.exceptions.RequestException as req_err:
        logging.error(f"An error occurred: {req_err}")
        return (
            jsonify(
                {
                    "error": "An error occurred",
                    "message": str(req_err),
                    "received_request": request.json,
                }
            ),
            500,
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return (
            jsonify(
                {
                    "error": "An unexpected error occurred",
                    "message": str(e),
                    "received_request": request.json,
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
