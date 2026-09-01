"""The browser UI's own layer (D11).

`session.py` decides a turn; `app.py` renders one. Nothing under here that a
test runs may import `streamlit` — see `session.py` for why that line is drawn
where it is, and `tests/test_ui_session.py` for the walk that enforces it.
"""
