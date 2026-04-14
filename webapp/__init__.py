"""Flask application factory."""
from flask import Flask
from webapp.routes import bp


def create_app() -> Flask:
    """Build and return the configured Flask app."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.register_blueprint(bp)
    return app
