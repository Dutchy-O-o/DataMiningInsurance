"""Entry point for the Smart Insurance Advisor web app.

Usage:
    python run.py              # Starts Flask on http://localhost:5000
"""
import os
import webbrowser
import threading

from webapp import create_app

app = create_app()


def _open_browser(port):
    webbrowser.open(f"http://localhost:{port}/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Smart Insurance Advisor V2.0\n  http://localhost:{port}\n")

    # Auto-open browser only when running locally (not in container).
    if os.environ.get("FLASK_ENV") != "production" and not os.environ.get("DOCKER"):
        threading.Timer(1.0, _open_browser, args=(port,)).start()

    app.run(host="0.0.0.0", port=port, debug=False)
