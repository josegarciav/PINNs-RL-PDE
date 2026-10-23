"""Entry point of ``pinnrl-dashboard``: open a browser tab and serve the Dash dashboard."""

import argparse
import contextlib
import io
import sys
import webbrowser

HOST = "127.0.0.1"


def bind_server(wsgi_app, port: int, attempts: int = 3):
    """Bind a werkzeug server for ``wsgi_app`` on the first free port from ``port``.

    The real server is bound, not a probe socket, so the port cannot be taken
    between the check and serving, and the check uses the server's own socket
    options (werkzeug sets ``SO_REUSEADDR``, so a port in ``TIME_WAIT`` counts
    as free).

    Args:
        wsgi_app: The WSGI application to serve (``app.server`` for Dash).
        port: Preferred port; ``port + 1, ...`` are tried next.
        attempts: How many ports to try.

    Returns:
        A bound, listening ``werkzeug.serving.BaseWSGIServer``; call
        ``serve_forever()`` on it.

    Raises:
        RuntimeError: If every candidate port is taken.
    """
    from werkzeug.serving import make_server

    for candidate in range(port, port + attempts):
        # On a bind error werkzeug prints advice to stderr and calls sys.exit(1).
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                return make_server(HOST, candidate, wsgi_app, threaded=True)
            except (OSError, SystemExit):
                pass
        print(f"Port {candidate} is in use.")
    raise RuntimeError(f"No free port in {port}-{port + attempts - 1}")


def run_dashboard(port=8050, open_browser=False):
    """Serve the Dash app on the first free port from ``port`` (blocking).

    Args:
        port: Preferred port; up to two higher ports are tried if it is taken.
        open_browser: Open the dashboard in a web browser once the server is listening.
    """
    try:
        from pinnrl.dashboard import app
    except ImportError as e:
        print(f"Error importing dashboard: {e}")
        print("Make sure all dependencies are installed (dash, plotly, etc.)")
        sys.exit(1)
    try:
        server = bind_server(app.server, port)
    except RuntimeError as e:
        print(f"{e}. Please close any running dashboards and try again.")
        sys.exit(1)

    url = f"http://{HOST}:{server.server_port}/"
    print(f"\nStarting PINNs-RL-PDE Dashboard on port {server.server_port}")
    print(f"Open {url} in your browser")
    if open_browser:
        webbrowser.open(url)
    app.enable_dev_tools(debug=False)  # what app.run() sets up before serving
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args():
    """Parse the ``pinnrl-dashboard`` command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PINNs-RL-PDE dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run the dashboard (training is launched from within):
    pinnrl-dashboard

  Specify dashboard port:
    pinnrl-dashboard --port 8051

  Serve without opening a browser tab:
    pinnrl-dashboard --no-browser

  Run headless training directly:
    pinnrl-train --pde "Heat Equation" --arch fourier --epochs 500
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the dashboard in a web browser",
    )
    return parser.parse_args()


def main():
    """Open ``http://127.0.0.1:<port>/`` (unless ``--no-browser``) and serve the dashboard."""
    args = parse_args()

    print("\n" + "=" * 50)
    print("PINNs-RL-PDE Framework")
    print("=" * 50 + "\n")

    run_dashboard(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
