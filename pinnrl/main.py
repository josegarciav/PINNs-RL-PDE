"""Entry point of ``pinnrl-dashboard``: open a browser tab and serve the Dash dashboard."""

import argparse
import sys
import webbrowser


def run_dashboard(port=8050):
    """Serve the Dash app on ``port`` (blocking).

    Tries the next port (three attempts in all) on an ``Exception`` whose
    message contains "Address already in use"; other errors exit with status 1.
    """
    try:
        from pinnrl.dashboard import app

        print(f"\nStarting PINNs-RL-PDE Dashboard on port {port}")
        print(f"Open http://127.0.0.1:{port}/ in your browser")

        max_retries = 3
        current_port = port

        for attempt in range(max_retries):
            try:
                app.run(debug=False, port=current_port)
                break
            except Exception as e:
                if "Address already in use" in str(e):
                    print(f"Port {current_port} is in use. Trying port {current_port + 1}...")
                    current_port += 1
                    if attempt == max_retries - 1:
                        print(f"Could not find available port after {max_retries} attempts.")
                        print("Please close any running dashboards and try again.")
                        sys.exit(1)
                else:
                    print(f"Error starting dashboard: {e}")
                    sys.exit(1)
    except ImportError as e:
        print(f"Error importing dashboard: {e}")
        print("Make sure all dependencies are installed (dash, plotly, etc.)")
        sys.exit(1)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)


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

    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}/")
    run_dashboard(args.port)


if __name__ == "__main__":
    main()
