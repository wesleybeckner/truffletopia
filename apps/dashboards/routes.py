"""Routes for parent Flask app."""
from flask import render_template, redirect, url_for
from flask import current_app as app


@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'index.jinja2',
        title='🍫 Truffletopia 🍭',
        description='Product Margin Analysis',
        template='home-template',
        body="This is a homepage served with Flask."
    )

@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))