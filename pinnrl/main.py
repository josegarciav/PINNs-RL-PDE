"""Entry point for the PINNs-RL-PDE framework."""

import argparse
import sys
import webbrowser


def run_dashboard(port=8050):
    """Run the dashboard module."""
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
                    print(f"Port {current_port} is in use. Trying port {current_port+1}...")
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
    """Parse command-line arguments for the main application."""
    parser = argparse.ArgumentParser(
        description="PINNs-RL-PDE Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run the dashboard (training is launched from within):
    pinnrl-dashboard

  Specify dashboard port:
    pinnrl-dashboard --port 8051

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
    """Entry point: launch the PINNs-RL-PDE dashboard."""
    args = parse_args()

    print("\n" + "=" * 50)
    print("PINNs-RL-PDE Framework")
    print("=" * 50 + "\n")

    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}/")
    run_dashboard(args.port)


if __name__ == "__main__":
    main()
