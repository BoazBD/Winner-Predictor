import requests
from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["POST"])
def proxy_request():
    target_url = request.json.get("url")
    headers = request.json.get(
        "headers", {}
    )  # Default to empty dict if headers not provided

    try:
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        return response.content
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
