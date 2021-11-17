"""Initialize Flask app."""
from flask import Flask

def init_app():
    """Construct core Flask application with embedded Dash app."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.DevConfig')

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes

        # Import Dash application
        from .margin_analysis.analyze import init_dashboard
        app = init_dashboard(app)

        from .margin_analysis.visualize import init_dashboard
        app = init_dashboard(app)

        from .forecast.train import init_dashboard
        app = init_dashboard(app)

        from .forecast.predict import init_dashboard
        app = init_dashboard(app)

        return app
