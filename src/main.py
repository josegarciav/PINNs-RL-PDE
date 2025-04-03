import os
import sys
import multiprocessing
import webbrowser
from time import sleep
import argparse


def run_interactive_trainer():
    """Run the interactive trainer module"""
    from src.interactive_trainer import main as trainer_main

    trainer_main()


def run_dashboard(port=8050):
    """Run the dashboard module"""
    from src.dashboard import app

    print(f"\n🚀 Starting PINNs-RL-PDE Dashboard on port {port}")
    print(f"📊 Open http://127.0.0.1:{port}/ in your browser")

    # Try different ports if the specified one is in use
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
                    print(
                        f"Could not find available port after {max_retries} attempts."
                    )
                    print("Please close any running dashboards and try again.")
                    sys.exit(1)
            else:
                print(f"Error starting dashboard: {e}")
                sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PINNs-RL-PDE Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run complete application (trainer first, then dashboard):
    python src/main.py
  
  Run only the interactive trainer:
    python src/main.py --trainer-only
    
  Run only the dashboard viewer:
    python src/main.py --dashboard-only
  
  Specify dashboard port:
    python src/main.py --port 8051
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Run only the dashboard viewer for existing experiments",
    )
    parser.add_argument(
        "--trainer-only",
        action="store_true",
        help="Run only the interactive trainer without dashboard",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Print welcome message
    print("\n" + "=" * 50)
    print("🧠 Welcome to PINNs-RL-PDE Framework")
    print("=" * 50 + "\n")

    if args.dashboard_only and args.trainer_only:
        print("Error: Cannot specify both --dashboard-only and --trainer-only")
        sys.exit(1)

    if args.dashboard_only:
        # Run only the dashboard viewer
        print("📊 Starting Dashboard Viewer...")
        run_dashboard(args.port)
    elif args.trainer_only:
        # Run only the interactive trainer
        print("🎯 Starting Interactive Trainer...")
        run_interactive_trainer()
    else:
        # Run complete application (trainer first, then dashboard)
        print("🎯 Starting Interactive Trainer...")
        trainer_process = multiprocessing.Process(target=run_interactive_trainer)
        trainer_process.start()

        # Wait a bit to let the trainer initialize
        sleep(2)

        print("\n📊 Starting Dashboard...")
        dashboard_process = multiprocessing.Process(
            target=run_dashboard, args=(args.port,)
        )
        dashboard_process.start()

        # Open the dashboard in the default web browser
        webbrowser.open(f"http://127.0.0.1:{args.port}/")

        try:
            # Wait for both processes to complete
            trainer_process.join()
            dashboard_process.join()
        except KeyboardInterrupt:
            print("\n\n⚠️  Received interrupt signal. Shutting down...")
            trainer_process.terminate()
            dashboard_process.terminate()
            trainer_process.join()
            dashboard_process.join()
            print("✅ Shutdown complete")
            sys.exit(0)


if __name__ == "__main__":
    main()
