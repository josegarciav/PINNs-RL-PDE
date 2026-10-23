"""Entry point of ``pinnrl-dashboard``: open a browser tab and serve the Dash dashboard."""

import argparse
import socket
import sys
import webbrowser


def find_free_port(port: int, attempts: int = 3) -> int:
    """Return the first of ``port, port + 1, ...`` (``attempts`` tries) free on 127.0.0.1.

    Probing with a socket is needed because the server cannot report it: on a
    busy port werkzeug calls ``sys.exit(1)`` instead of raising.

    Raises:
        RuntimeError: If every candidate port is taken.
    """
    for candidate in range(port, port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                print(f"Port {candidate} is in use.")
                continue
        return candidate
    raise RuntimeError(f"No free port in {port}-{port + attempts - 1}")


def run_dashboard(port=8050, open_browser=False):
    """Serve the Dash app on the first free port from ``port`` (blocking).

    Args:
        port: Preferred port; up to two higher ports are tried if it is taken.
        open_browser: Open the dashboard in a web browser once the port is known.
    """
    try:
        from pinnrl.dashboard import app
    except ImportError as e:
        print(f"Error importing dashboard: {e}")
        print("Make sure all dependencies are installed (dash, plotly, etc.)")
        sys.exit(1)
    try:
        port = find_free_port(port)
    except RuntimeError as e:
        print(f"{e}. Please close any running dashboards and try again.")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    print(f"\nStarting PINNs-RL-PDE Dashboard on port {port}")
    print(f"Open {url} in your browser")
    if open_browser:
        webbrowser.open(url)
    app.run(debug=False, port=port)


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
