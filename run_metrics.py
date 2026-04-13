"""
run_metrics.py  -  Edge Computing Metrics Monitor
==================================================
Run this in a SEPARATE terminal alongside main.py.

Usage examples
--------------
# Auto-discover main.py process and sample every 1 second:
    python run_metrics.py

# Attach to a specific PID:
    python run_metrics.py --pid 1234

# Custom interval and report directory:
    python run_metrics.py --interval 2.0 --report-dir ./reports

# Run for a fixed duration then save:
    python run_metrics.py --duration 60

Press Ctrl+C at any time to stop and save the report.

Metrics collected
-----------------
  System  : CPU % (overall + per-core snapshot), RAM MB/%, CPU temperature
  Process : Target process CPU %, RSS RAM MB, thread count
  Edge KPIs: sustained load index, RAM/core, peaks
"""

from __future__ import division

import argparse
import os
import sys
import time

try:
    import psutil
except ImportError:
    print("[ERROR] psutil is required. Install it: pip install psutil", flush=True)
    sys.exit(1)

from metrics import SystemMetricsMonitor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge Computing Metrics Monitor — run alongside main.py"
    )
    parser.add_argument(
        "--pid", type=int, default=None,
        help="PID of the target process to monitor (default: auto-discover main.py)"
    )
    parser.add_argument(
        "--name", type=str, default="main.py",
        help="Process name/cmdline substring to auto-discover (default: main.py)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Sample interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--report-dir", type=str, default="reports",
        help="Directory to write reports to (default: reports)"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Stop automatically after this many seconds (default: run until Ctrl+C)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sample live output; only print summary at end"
    )
    return parser.parse_args()


def print_per_core(interval=0.5):
    """Print a one-time per-core CPU snapshot."""
    cores = psutil.cpu_percent(percpu=True, interval=interval)
    print("\n  Per-core CPU snapshot:", flush=True)
    for i, pct in enumerate(cores):
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print("    Core {:2d}: [{}] {:5.1f}%".format(i, bar, pct), flush=True)
    print("", flush=True)


def run():
    args = parse_args()

    monitor = SystemMetricsMonitor(
        target_pid=args.pid,
        target_name=args.name,
        sample_interval_sec=args.interval,
        report_dir=args.report_dir,
    )

    SEP = "-" * 64
    print(SEP, flush=True)
    print("  Edge Metrics Monitor  |  interval={:.1f}s  |  Ctrl+C to stop & save".format(
        args.interval), flush=True)
    print(SEP, flush=True)

    # One-time per-core snapshot at startup
    print_per_core(interval=min(args.interval, 1.0))

    deadline = None
    if args.duration is not None:
        deadline = time.time() + args.duration

    try:
        while True:
            monitor.sample()

            if not args.quiet:
                monitor.print_live()

            if deadline is not None and time.time() >= deadline:
                print("\n[METRICS] Duration reached ({:.0f}s). Saving report...".format(
                    args.duration), flush=True)
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[METRICS] Interrupted. Saving report...", flush=True)

    # Final per-core snapshot
    print_per_core(interval=0.5)

    monitor.save_report()


if __name__ == "__main__":
    run()
