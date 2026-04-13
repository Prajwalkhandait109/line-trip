"""
Edge Computing Metrics Monitor
==============================
External process/system monitor for the YOLOv5 line-crossing counter.
Run this file separately alongside main.py in a second terminal.

Metrics collected (all sampled externally via psutil):
  System  : CPU % (overall + per-core), RAM MB/%, CPU temperature
  Process : target process CPU %, RSS RAM MB, thread count, uptime
  Edge KPIs: efficiency score, MB-per-core, sustained load index
"""

from __future__ import division

import csv
import json
import os
import statistics
import time
from collections import deque
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _pct_bar(pct, width=20):
    """Return a simple ASCII progress bar for a percentage value."""
    filled = int(round(pct / 100.0 * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "] {:5.1f}%".format(pct)


class SystemMetricsMonitor(object):
    """
    Samples system and per-process metrics at a configurable interval.
    Designed to run in a separate process/terminal alongside main.py.
    """

    MAX_SAMPLES = 50000

    def __init__(self, target_pid=None, target_name="main.py",
                 sample_interval_sec=1.0, report_dir="reports"):
        if not PSUTIL_AVAILABLE:
            raise RuntimeError(
                "psutil is required. Install it: pip install psutil"
            )

        self.target_pid = target_pid
        self.target_name = target_name
        self.sample_interval_sec = float(sample_interval_sec)
        self.report_dir = report_dir

        self.session_start_ts = time.time()
        self.session_start_dt = datetime.now()

        # ── Sample buffers ────────────────────────────────────────────────
        self._sys_cpu_pct    = deque(maxlen=self.MAX_SAMPLES)
        self._sys_ram_mb     = deque(maxlen=self.MAX_SAMPLES)
        self._sys_ram_pct    = deque(maxlen=self.MAX_SAMPLES)
        self._sys_temp_c     = deque(maxlen=self.MAX_SAMPLES)
        self._proc_cpu_pct   = deque(maxlen=self.MAX_SAMPLES)
        self._proc_ram_mb    = deque(maxlen=self.MAX_SAMPLES)
        self._proc_threads   = deque(maxlen=self.MAX_SAMPLES)
        self._timestamps     = deque(maxlen=self.MAX_SAMPLES)

        self._samples_taken  = 0
        self._target_proc    = None   # psutil.Process handle for target
        self._cpu_count      = psutil.cpu_count(logical=True) or 1
        self._cpu_count_phys = psutil.cpu_count(logical=False) or 1

        os.makedirs(report_dir, exist_ok=True)

        # Warm-up: first cpu_percent call always returns 0.0
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(percpu=True, interval=None)

        self._attach_target()

    # ── Target process attachment ─────────────────────────────────────────

    def _attach_target(self):
        """Find and attach to the target process."""
        if self.target_pid is not None:
            try:
                self._target_proc = psutil.Process(self.target_pid)
                self._target_proc.cpu_percent(interval=None)  # warm-up
                print("[METRICS] Attached to PID {} ({})".format(
                    self.target_pid,
                    self._target_proc.name()), flush=True)
            except psutil.NoSuchProcess:
                print("[METRICS] PID {} not found — monitoring system only.".format(
                    self.target_pid), flush=True)
                self._target_proc = None
        else:
            self._target_proc = self._discover_target()

    def _discover_target(self):
        """Search running processes for one matching target_name."""
        candidates = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if self.target_name in cmdline:
                    candidates.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not candidates:
            print("[METRICS] '{}' not found — monitoring system only.".format(
                self.target_name), flush=True)
            return None
        proc = candidates[0]
        try:
            proc.cpu_percent(interval=None)   # warm-up
            print("[METRICS] Auto-attached to PID {} — {}".format(
                proc.pid, " ".join(proc.cmdline())), flush=True)
        except Exception:
            pass
        return proc

    # ── Single sample ─────────────────────────────────────────────────────

    def sample(self):
        """Take one metrics snapshot and append to buffers."""
        ts = time.time()

        # System CPU
        sys_cpu = psutil.cpu_percent(interval=None)
        self._sys_cpu_pct.append(sys_cpu)

        # System RAM
        vm = psutil.virtual_memory()
        self._sys_ram_mb.append(vm.used / 1024.0 / 1024.0)
        self._sys_ram_pct.append(vm.percent)

        # CPU temperature
        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ("cpu_thermal", "cpu-thermal", "coretemp",
                            "k10temp", "acpitz", "soc_thermal"):
                    if key in temps and temps[key]:
                        temp = temps[key][0].current
                        break
                if temp is None:
                    first = next(iter(temps))
                    if temps[first]:
                        temp = temps[first][0].current
        except (AttributeError, Exception):
            pass
        if temp is not None:
            self._sys_temp_c.append(temp)

        # Per-process
        if self._target_proc is not None:
            try:
                self._proc_cpu_pct.append(self._target_proc.cpu_percent(interval=None))
                mem = self._target_proc.memory_info()
                self._proc_ram_mb.append(mem.rss / 1024.0 / 1024.0)
                self._proc_threads.append(self._target_proc.num_threads())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._target_proc = None   # process exited
                print("[METRICS] Target process ended.", flush=True)

        self._timestamps.append(ts)
        self._samples_taken += 1
        return sys_cpu, vm.percent, temp

    # ── Live display ──────────────────────────────────────────────────────

    def print_live(self):
        """Print a compact live-update line with current metrics."""
        elapsed = time.time() - self.session_start_ts
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        cpu = self._sys_cpu_pct[-1] if self._sys_cpu_pct else 0.0
        ram_pct = self._sys_ram_pct[-1] if self._sys_ram_pct else 0.0
        ram_mb  = self._sys_ram_mb[-1] if self._sys_ram_mb else 0.0
        temp    = self._sys_temp_c[-1] if self._sys_temp_c else None
        p_cpu   = self._proc_cpu_pct[-1] if self._proc_cpu_pct else None
        p_ram   = self._proc_ram_mb[-1] if self._proc_ram_mb else None
        threads = self._proc_threads[-1] if self._proc_threads else None

        line = ("[{:02d}:{:02d}:{:02d}]  "
                "SysCPU {bar_cpu}  "
                "RAM {ram_mb:.0f}MB {ram_pct:.1f}%"
                ).format(h, m, s,
                         bar_cpu=_pct_bar(cpu, 16),
                         ram_mb=ram_mb, ram_pct=ram_pct)
        if temp is not None:
            line += "  Temp {:5.1f}C".format(temp)
        if p_cpu is not None:
            line += "  | Proc CPU {:5.1f}%  RAM {:.0f}MB".format(p_cpu, p_ram or 0)
        if threads is not None:
            line += "  thr={}".format(threads)
        print(line, flush=True)

    # ── Stats helper ──────────────────────────────────────────────────────

    @staticmethod
    def _stats(data):
        if not data:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                    "p50": 0.0, "p95": 0.0, "stdev": 0.0}
        lst = sorted(data)
        n = len(lst)
        return {
            "count": n,
            "mean":  round(statistics.mean(lst), 3),
            "min":   round(lst[0], 3),
            "max":   round(lst[-1], 3),
            "p50":   round(lst[max(0, int(n * 0.50) - 1)], 3),
            "p95":   round(lst[max(0, int(n * 0.95) - 1)], 3),
            "stdev": round(statistics.stdev(lst), 3) if n > 1 else 0.0,
        }

    # ── Build report ──────────────────────────────────────────────────────

    def build_report(self):
        elapsed = max(time.time() - self.session_start_ts, 1e-6)
        st = self._stats

        # Edge efficiency
        mean_cpu = st(self._sys_cpu_pct)["mean"]
        mean_ram_mb = st(self._sys_ram_mb)["mean"]
        ram_total_mb = psutil.virtual_memory().total / 1024.0 / 1024.0
        # Sustained load index: avg CPU / cores (lower = less stressed per core)
        sustained_load_idx = round(mean_cpu / max(self._cpu_count, 1), 3)
        # MB per logical core
        ram_per_core = round(mean_ram_mb / max(self._cpu_count, 1), 2)

        target_info = {}
        if self._target_proc is not None:
            try:
                target_info = {
                    "pid":  self._target_proc.pid,
                    "name": self._target_proc.name(),
                    "exe":  self._target_proc.exe(),
                }
            except Exception:
                pass
        elif self.target_pid:
            target_info = {"pid": self.target_pid, "status": "exited"}

        return {
            "session": {
                "start_time":       self.session_start_dt.isoformat(timespec="seconds"),
                "duration_sec":     round(elapsed, 2),
                "samples_taken":    self._samples_taken,
                "sample_interval":  self.sample_interval_sec,
                "logical_cores":    self._cpu_count,
                "physical_cores":   self._cpu_count_phys,
                "total_ram_mb":     round(ram_total_mb, 1),
                "target_process":   target_info,
            },
            "system_cpu_pct":  st(self._sys_cpu_pct),
            "system_ram_mb":   st(self._sys_ram_mb),
            "system_ram_pct":  st(self._sys_ram_pct),
            "cpu_temp_c":      st(self._sys_temp_c),
            "process_cpu_pct": st(self._proc_cpu_pct),
            "process_ram_mb":  st(self._proc_ram_mb),
            "process_threads": st(self._proc_threads),
            "edge_efficiency": {
                "sustained_load_index":  sustained_load_idx,
                "ram_mb_per_core":       ram_per_core,
                "peak_cpu_pct":          round(max(self._sys_cpu_pct, default=0.0), 2),
                "peak_ram_mb":           round(max(self._sys_ram_mb, default=0.0), 2),
                "peak_temp_c":           round(max(self._sys_temp_c, default=0.0), 2) if self._sys_temp_c else None,
                "peak_proc_cpu_pct":     round(max(self._proc_cpu_pct, default=0.0), 2) if self._proc_cpu_pct else None,
                "peak_proc_ram_mb":      round(max(self._proc_ram_mb, default=0.0), 2) if self._proc_ram_mb else None,
            },
        }

    # ── Save & print ──────────────────────────────────────────────────────

    def save_report(self, prefix="metrics"):
        ts = self.session_start_dt.strftime("%Y%m%d_%H%M%S")
        report = self.build_report()

        json_path = os.path.join(self.report_dir, "{}_{}.json".format(prefix, ts))
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # CSV time-series
        csv_path = os.path.join(self.report_dir, "{}_{}_timeseries.csv".format(prefix, ts))
        rows = zip(
            self._timestamps,
            self._sys_cpu_pct,
            self._sys_ram_mb,
            self._sys_ram_pct,
            list(self._sys_temp_c) + [None] * max(0, self._samples_taken - len(self._sys_temp_c)),
            list(self._proc_cpu_pct) + [None] * max(0, self._samples_taken - len(self._proc_cpu_pct)),
            list(self._proc_ram_mb) + [None] * max(0, self._samples_taken - len(self._proc_ram_mb)),
            list(self._proc_threads) + [None] * max(0, self._samples_taken - len(self._proc_threads)),
        )
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "sys_cpu_pct", "sys_ram_mb", "sys_ram_pct",
                        "temp_c", "proc_cpu_pct", "proc_ram_mb", "proc_threads"])
            for row in rows:
                w.writerow(["" if v is None else v for v in row])

        self.print_summary(report)
        print("\n[METRICS] JSON report    : {}".format(json_path), flush=True)
        print("[METRICS] Time-series CSV: {}".format(csv_path), flush=True)
        return json_path

    def print_summary(self, report=None):
        if report is None:
            report = self.build_report()

        SEP  = "=" * 64
        SEP2 = "-" * 64
        sess = report["session"]

        print("\n" + SEP)
        print("  EDGE COMPUTING PERFORMANCE REPORT")
        print("  Started  : {}   Duration : {:.1f} s".format(
            sess["start_time"], sess["duration_sec"]))
        print("  Samples  : {}   Interval : {} s".format(
            sess["samples_taken"], sess["sample_interval"]))
        print(SEP)

        def _row(label, s, unit=""):
            if s["count"] == 0:
                print("  {:28s}: no data".format(label))
                return
            print(("  {:28s}: mean={:.2f}{}  p50={:.2f}{}  p95={:.2f}{}  "
                   "min={:.2f}{}  max={:.2f}{}").format(
                label,
                s["mean"], unit, s["p50"], unit, s["p95"], unit,
                s["min"], unit, s["max"], unit))

        print("\n SYSTEM CPU")
        print(SEP2)
        print("  Logical cores           : {:>4d}  (physical: {})".format(
            sess["logical_cores"], sess["physical_cores"]))
        _row("System CPU %",  report["system_cpu_pct"],  "%")

        print("\n MEMORY")
        print(SEP2)
        print("  Total RAM               : {:.0f} MB".format(sess["total_ram_mb"]))
        _row("System RAM used", report["system_ram_mb"],  " MB")
        _row("System RAM %",    report["system_ram_pct"], "%")

        if report["cpu_temp_c"]["count"] > 0:
            print("\n CPU TEMPERATURE")
            print(SEP2)
            _row("CPU temp", report["cpu_temp_c"], " C")

        if report["process_cpu_pct"]["count"] > 0:
            print("\n TARGET PROCESS  ({})".format(
                sess.get("target_process", {}).get("pid", "??")))
            print(SEP2)
            _row("Process CPU %",  report["process_cpu_pct"], "%")
            _row("Process RAM",     report["process_ram_mb"],  " MB")
            _row("Thread count",    report["process_threads"], "")
        else:
            print("\n TARGET PROCESS  : not monitored (no PID found)")

        ee = report["edge_efficiency"]
        print("\n EDGE EFFICIENCY")
        print(SEP2)
        print("  Peak system CPU         : {:>7.2f}%".format(ee["peak_cpu_pct"]))
        print("  Peak system RAM         : {:>7.0f} MB".format(ee["peak_ram_mb"]))
        if ee.get("peak_temp_c") is not None:
            print("  Peak CPU temperature    : {:>7.1f} C".format(ee["peak_temp_c"]))
        print("  Sustained load index    : {:>7.3f}  (avg CPU / cores, lower=better)".format(
            ee["sustained_load_index"]))
        print("  RAM per core            : {:>7.1f} MB".format(ee["ram_mb_per_core"]))
        if ee.get("peak_proc_cpu_pct") is not None:
            print("  Peak process CPU        : {:>7.2f}%".format(ee["peak_proc_cpu_pct"]))
        if ee.get("peak_proc_ram_mb") is not None:
            print("  Peak process RAM        : {:>7.0f} MB".format(ee["peak_proc_ram_mb"]))

        print("\n" + SEP + "\n")


from __future__ import division

import csv
import json
import os
import statistics
import time
from collections import deque
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stats(data):
    """Return a summary-statistics dict for an iterable of floats."""
    if not data:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "stdev": 0.0}
    lst = sorted(data)
    n = len(lst)
    p50_i = max(0, int(n * 0.50) - 1)
    p95_i = max(0, int(n * 0.95) - 1)
    return {
        "count": n,
        "mean": round(statistics.mean(lst), 4),
        "min": round(lst[0], 4),
        "max": round(lst[-1], 4),
        "p50": round(lst[p50_i], 4),
        "p95": round(lst[p95_i], 4),
        "stdev": round(statistics.stdev(lst), 4) if n > 1 else 0.0,
    }


def _fmt(v, unit="", width=8, decimals=2):
    if v is None:
        return "{:>{w}}".format("N/A", w=width)
    fmt = "{:>{w}.{d}f}{}".format(v, unit, w=width, d=decimals)
    return fmt


# ──────────────────────────────────────────────────────────────────────────────
# Main collector
# ──────────────────────────────────────────────────────────────────────────────

class MetricsCollector(object):
    """Collects and reports edge-computing performance metrics."""

    MAX_SAMPLES = 20000   # cap per-deque memory

    def __init__(self, report_dir="reports", report_interval_sec=5.0,
                 model_path=None, settings=None):
        self.report_dir = report_dir
        self.report_interval_sec = float(report_interval_sec)
        self.model_path = model_path
        self.settings = settings

        self.session_start_ts = time.time()
        self.session_start_dt = datetime.now()

        # ── Inference timing (ms) ──────────────────────────────────────────
        self._pre_ms    = deque(maxlen=self.MAX_SAMPLES)
        self._fwd_ms    = deque(maxlen=self.MAX_SAMPLES)
        self._post_ms   = deque(maxlen=self.MAX_SAMPLES)
        self._total_ms  = deque(maxlen=self.MAX_SAMPLES)

        # ── FPS / Throughput ──────────────────────────────────────────────
        self._display_fps   = deque(maxlen=self.MAX_SAMPLES)
        self._inf_fps       = deque(maxlen=self.MAX_SAMPLES)
        self._last_inf_time = 0.0          # for computing inference FPS
        self._inf_fps_ema   = 0.0

        # ── End-to-end latency (ms) ───────────────────────────────────────
        self._latency_ms    = deque(maxlen=self.MAX_SAMPLES)
        self._submit_ts     = 0.0          # set by mark_frame_submitted()

        # ── Detection quality ──────────────────────────────────────────────
        self._det_counts    = deque(maxlen=self.MAX_SAMPLES)
        self._confidences   = deque(maxlen=self.MAX_SAMPLES)

        # ── Crossing counters ──────────────────────────────────────────────
        self.crossing_in  = 0
        self.crossing_out = 0

        # ── Frame counters ─────────────────────────────────────────────────
        self.total_frames_received  = 0
        self.total_frames_processed = 0

        # ── System resource snapshots ─────────────────────────────────────
        self._sys_snapshots    = []
        self._last_snap_time   = 0.0
        self._peak_ram_mb      = 0.0
        self._peak_cpu_pct     = 0.0

        # psutil process handle
        self._proc = None
        if PSUTIL_AVAILABLE:
            try:
                self._proc = psutil.Process()
                self._proc.cpu_percent(interval=None)   # warm-up call
                psutil.cpu_percent(interval=None)       # warm-up system CPU
            except Exception:
                pass

        os.makedirs(report_dir, exist_ok=True)

    # ── Public record API ─────────────────────────────────────────────────────

    def mark_frame_submitted(self):
        """Call immediately before submitting a frame to the inference worker."""
        self._submit_ts = time.perf_counter()

    def record_frame_received(self):
        """Increment the raw frame counter (every frame the main loop sees)."""
        self.total_frames_received += 1

    def record_inference(self, preprocess_ms, forward_ms, postprocess_ms,
                         num_detections, confidences=None):
        """Call once per completed inference cycle (inside the result-ready block)."""
        total_ms = preprocess_ms + forward_ms + postprocess_ms
        self._pre_ms.append(preprocess_ms)
        self._fwd_ms.append(forward_ms)
        self._post_ms.append(postprocess_ms)
        self._total_ms.append(total_ms)

        self._det_counts.append(num_detections)
        if confidences:
            self._confidences.extend(confidences)

        # End-to-end latency
        if self._submit_ts > 0:
            latency_ms = (time.perf_counter() - self._submit_ts) * 1000.0
            self._latency_ms.append(latency_ms)

        # Inference FPS (EMA of inter-call rate)
        now = time.perf_counter()
        if self._last_inf_time > 0:
            dt = max(now - self._last_inf_time, 1e-6)
            inst = 1.0 / dt
            if self._inf_fps_ema == 0.0:
                self._inf_fps_ema = inst
            else:
                self._inf_fps_ema = 0.85 * self._inf_fps_ema + 0.15 * inst
            self._inf_fps.append(self._inf_fps_ema)
        self._last_inf_time = now

        self.total_frames_processed += 1

        # Periodic system resource snapshot
        if (now - self._last_snap_time) >= self.report_interval_sec:
            self._take_sys_snapshot()
            self._last_snap_time = now

    def record_display_fps(self, fps):
        """Call with the smoothed display-loop FPS each frame."""
        if fps > 0:
            self._display_fps.append(fps)

    def record_crossing(self, direction):
        """Call for each line-crossing event ('IN' or 'OUT')."""
        if direction == "IN":
            self.crossing_in += 1
        elif direction == "OUT":
            self.crossing_out += 1

    # ── System snapshot ───────────────────────────────────────────────────────

    def _take_sys_snapshot(self):
        if not PSUTIL_AVAILABLE:
            return
        try:
            cpu_pct  = psutil.cpu_percent(interval=None)
            cpu_each = psutil.cpu_percent(percpu=True, interval=None)
            vm       = psutil.virtual_memory()
            ram_mb   = vm.used / 1024.0 / 1024.0
            ram_pct  = vm.percent

            # CPU temperature
            temp_c = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for key in ("cpu_thermal", "cpu-thermal", "coretemp",
                                "k10temp", "acpitz", "soc_thermal"):
                        if key in temps and temps[key]:
                            temp_c = temps[key][0].current
                            break
                    if temp_c is None:
                        first = next(iter(temps))
                        if temps[first]:
                            temp_c = temps[first][0].current
            except (AttributeError, Exception):
                pass

            # Per-process stats
            proc_cpu_pct = None
            proc_ram_mb  = None
            try:
                if self._proc is not None:
                    proc_cpu_pct = self._proc.cpu_percent(interval=None)
                    proc_ram_mb  = self._proc.memory_info().rss / 1024.0 / 1024.0
            except Exception:
                pass

            snap = {
                "ts":           round(time.time(), 3),
                "cpu_pct":      cpu_pct,
                "cpu_each":     cpu_each,
                "ram_mb":       round(ram_mb, 2),
                "ram_pct":      ram_pct,
                "temp_c":       temp_c,
                "proc_cpu_pct": proc_cpu_pct,
                "proc_ram_mb":  round(proc_ram_mb, 2) if proc_ram_mb else None,
            }
            self._sys_snapshots.append(snap)
            self._peak_ram_mb  = max(self._peak_ram_mb, ram_mb)
            self._peak_cpu_pct = max(self._peak_cpu_pct, cpu_pct)
        except Exception:
            pass

    # ── Report builder ────────────────────────────────────────────────────────

    def build_report(self):
        """Compile all collected metrics into a serialisable dict."""
        # Final snapshot
        self._take_sys_snapshot()

        elapsed = max(time.time() - self.session_start_ts, 1e-6)
        cpu_cores = os.cpu_count() or 1

        model_size_mb = None
        if self.model_path and os.path.exists(self.model_path):
            model_size_mb = round(os.path.getsize(self.model_path) / 1024.0 / 1024.0, 3)

        # Frame drop rate
        drop_rate = 0.0
        if self.total_frames_received > 0:
            drop_rate = 1.0 - (
                self.total_frames_processed / float(self.total_frames_received)
            )

        # Detection stats
        avg_conf = 0.0
        if self._confidences:
            avg_conf = statistics.mean(self._confidences)
        det_rate = 0.0
        if self._det_counts:
            det_rate = sum(1 for c in self._det_counts if c > 0) / float(len(self._det_counts))

        # System averages
        sys_agg = {}
        if self._sys_snapshots:
            def _avg(key):
                vals = [s[key] for s in self._sys_snapshots if s.get(key) is not None]
                return round(statistics.mean(vals), 3) if vals else None

            sys_agg = {
                "avg_cpu_pct":       _avg("cpu_pct"),
                "avg_ram_mb":        _avg("ram_mb"),
                "avg_ram_pct":       _avg("ram_pct"),
                "avg_temp_c":        _avg("temp_c"),
                "max_temp_c":        max((s["temp_c"] for s in self._sys_snapshots
                                         if s.get("temp_c") is not None), default=None),
                "avg_proc_cpu_pct":  _avg("proc_cpu_pct"),
                "avg_proc_ram_mb":   _avg("proc_ram_mb"),
                "peak_ram_mb":       round(self._peak_ram_mb, 2),
                "peak_cpu_pct":      round(self._peak_cpu_pct, 2),
                "snapshots_taken":   len(self._sys_snapshots),
            }

        # Settings summary
        settings_summary = {}
        if self.settings is not None:
            s = self.settings
            settings_summary = {
                "model_path":          getattr(s, "model_path", None),
                "model_input_size":    getattr(s, "model_input_size", None),
                "confidence_threshold": getattr(s, "confidence_threshold", None),
                "nms_threshold":       getattr(s, "nms_threshold", None),
                "frame_width":         getattr(s, "frame_width", None),
                "skip_frames":         getattr(s, "skip_frames", None),
                "cpu_threads":         getattr(s, "cpu_threads", None),
                "use_tracking":        getattr(s, "use_tracking", None),
            }

        inf_total_stats = _stats(self._total_ms)
        mean_inf_ms = inf_total_stats["mean"]

        report = {
            "session": {
                "start_time":      self.session_start_dt.isoformat(timespec="seconds"),
                "duration_sec":    round(elapsed, 2),
                "cpu_cores":       cpu_cores,
                "model_path":      self.model_path,
                "model_size_mb":   model_size_mb,
                "psutil_available": PSUTIL_AVAILABLE,
                "settings":        settings_summary,
            },
            "throughput": {
                "total_frames_received":  self.total_frames_received,
                "total_frames_processed": self.total_frames_processed,
                "frame_drop_rate_pct":    round(drop_rate * 100.0, 2),
                "display_fps":            _stats(self._display_fps),
                "inference_fps":          _stats(self._inf_fps),
            },
            "inference_timing_ms": {
                "preprocess":   _stats(self._pre_ms),
                "forward_pass": _stats(self._fwd_ms),
                "postprocess":  _stats(self._post_ms),
                "total":        inf_total_stats,
            },
            "latency_ms": _stats(self._latency_ms),
            "detection_quality": {
                "avg_detections_per_frame": round(
                    statistics.mean(self._det_counts), 3) if self._det_counts else 0.0,
                "avg_confidence":     round(avg_conf, 4),
                "detection_rate_pct": round(det_rate * 100.0, 2),
                "total_crossings_in":  self.crossing_in,
                "total_crossings_out": self.crossing_out,
            },
            "system_resources": sys_agg,
            "edge_efficiency": {
                "inference_ms_per_core":       round(mean_inf_ms / cpu_cores, 3),
                "frames_processed_per_minute": round(
                    self.total_frames_processed / max(elapsed / 60.0, 1e-9), 2),
                "throughput_efficiency_pct":   round((1.0 - drop_rate) * 100.0, 2),
                # Theoretical max FPS if inference were the only bottleneck
                "theoretical_max_fps": round(
                    1000.0 / mean_inf_ms, 2) if mean_inf_ms > 0 else None,
            },
        }
        return report

    # ── Output ────────────────────────────────────────────────────────────────

    def save_report(self, prefix="metrics"):
        """Save JSON report + system CSV, then print terminal summary."""
        ts = self.session_start_dt.strftime("%Y%m%d_%H%M%S")
        report = self.build_report()

        json_path = os.path.join(self.report_dir, "{}_{}.json".format(prefix, ts))
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        csv_path = None
        if self._sys_snapshots:
            csv_path = os.path.join(
                self.report_dir, "{}_{}_syslog.csv".format(prefix, ts)
            )
            fieldnames = [
                "ts", "cpu_pct", "ram_mb", "ram_pct",
                "temp_c", "proc_cpu_pct", "proc_ram_mb",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for snap in self._sys_snapshots:
                    writer.writerow({k: snap.get(k, "") for k in fieldnames})

        self.print_summary(report)
        print("\n[METRICS] JSON report : {}".format(json_path), flush=True)
        if csv_path:
            print("[METRICS] System CSV  : {}".format(csv_path), flush=True)
        return json_path

    def print_summary(self, report=None):
        """Pretty-print the metrics table to stdout."""
        if report is None:
            report = self.build_report()

        SEP  = "=" * 64
        SEP2 = "-" * 64

        print("\n" + SEP)
        print("  EDGE COMPUTING PERFORMANCE REPORT")
        print("  Started : {}   Duration : {:.1f} s".format(
            report["session"]["start_time"],
            report["session"]["duration_sec"]))
        print(SEP)

        # ── Throughput ────────────────────────────────────────────────────
        t = report["throughput"]
        print("\n THROUGHPUT")
        print(SEP2)
        print("  Frames received         : {:>8d}".format(t["total_frames_received"]))
        print("  Frames processed        : {:>8d}   (drop {:>5.1f}%)".format(
            t["total_frames_processed"], t["frame_drop_rate_pct"]))
        print("  Display FPS  avg/p95    : {:>8.2f}  / {:.2f}".format(
            t["display_fps"]["mean"], t["display_fps"]["p95"]))
        print("  Inference FPS avg/p95   : {:>8.2f}  / {:.2f}".format(
            t["inference_fps"]["mean"], t["inference_fps"]["p95"]))

        # ── Inference timing ──────────────────────────────────────────────
        it = report["inference_timing_ms"]
        print("\n INFERENCE TIMING  (milliseconds)")
        print(SEP2)
        print("  {:20s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
            "Stage", "mean", "p50", "p95", "stdev"))
        print("  " + "-"*60)
        for label, key in [("Preprocess",   "preprocess"),
                            ("Forward pass", "forward_pass"),
                            ("Postprocess",  "postprocess"),
                            ("TOTAL",        "total")]:
            s = it[key]
            print("  {:20s}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}".format(
                label, s["mean"], s["p50"], s["p95"], s["stdev"]))

        # ── Latency ───────────────────────────────────────────────────────
        lat = report["latency_ms"]
        print("\n LATENCY  (end-to-end: frame submit → result ready, ms)")
        print(SEP2)
        print("  Mean                    : {:>8.2f} ms".format(lat["mean"]))
        print("  P50 (median)            : {:>8.2f} ms".format(lat["p50"]))
        print("  P95                     : {:>8.2f} ms".format(lat["p95"]))
        print("  Jitter (stdev)          : {:>8.2f} ms".format(lat["stdev"]))
        print("  Best                    : {:>8.2f} ms".format(lat["min"]))
        print("  Worst                   : {:>8.2f} ms".format(lat["max"]))

        # ── Detection quality / accuracy ──────────────────────────────────
        dq = report["detection_quality"]
        print("\n DETECTION QUALITY  (accuracy proxy)")
        print(SEP2)
        print("  Avg detections / frame  : {:>8.2f}".format(dq["avg_detections_per_frame"]))
        print("  Avg detection confidence: {:>8.4f}  (higher = more certain)".format(
            dq["avg_confidence"]))
        print("  Detection rate          : {:>8.1f}%  (frames with >= 1 detection)".format(
            dq["detection_rate_pct"]))
        print("  Crossings IN            : {:>8d}".format(dq["total_crossings_in"]))
        print("  Crossings OUT           : {:>8d}".format(dq["total_crossings_out"]))

        # ── System resources ──────────────────────────────────────────────
        sr = report.get("system_resources", {})
        if sr:
            print("\n SYSTEM RESOURCES")
            print(SEP2)
            if not PSUTIL_AVAILABLE:
                print("  [!] psutil not installed – install it for resource metrics")
            else:
                def _row(label, key, unit="", decimals=1):
                    v = sr.get(key)
                    if v is not None:
                        print("  {:28s}: {:>8.{d}f}{}".format(
                            label, v, unit, d=decimals))

                _row("Avg system CPU",      "avg_cpu_pct",      "%")
                _row("Peak system CPU",     "peak_cpu_pct",     "%")
                _row("Avg RAM used",        "avg_ram_mb",       " MB")
                _row("Peak RAM used",       "peak_ram_mb",      " MB")
                _row("Avg RAM usage",       "avg_ram_pct",      "%")
                _row("Avg CPU temperature", "avg_temp_c",       " °C")
                _row("Max CPU temperature", "max_temp_c",       " °C")
                _row("Avg process CPU",     "avg_proc_cpu_pct", "%")
                _row("Avg process RAM",     "avg_proc_ram_mb",  " MB")
                print("  {:28s}: {:>8d}".format(
                    "Snapshots taken", sr.get("snapshots_taken", 0)))

        # ── Model info ────────────────────────────────────────────────────
        sess = report["session"]
        print("\n MODEL")
        print(SEP2)
        if sess.get("model_size_mb"):
            print("  Size on disk            : {:>8.2f} MB".format(sess["model_size_mb"]))
        if sess.get("model_path"):
            print("  Path                    : {}".format(sess["model_path"]))
        print("  CPU cores available     : {:>8d}".format(sess["cpu_cores"]))
        cfg = sess.get("settings", {})
        if cfg:
            print("  Input size              : {:>8}".format(cfg.get("model_input_size", "?")))
            print("  Confidence threshold    : {:>8}".format(cfg.get("confidence_threshold", "?")))
            print("  CPU threads (ONNX)      : {:>8}".format(cfg.get("cpu_threads", "?")))

        # ── Edge efficiency ───────────────────────────────────────────────
        ee = report["edge_efficiency"]
        print("\n EDGE EFFICIENCY")
        print(SEP2)
        print("  Inference ms per core   : {:>8.3f} ms".format(ee["inference_ms_per_core"]))
        print("  Frames processed/minute : {:>8.2f}".format(ee["frames_processed_per_minute"]))
        print("  Throughput efficiency   : {:>8.1f}%  (processed / received)".format(
            ee["throughput_efficiency_pct"]))
        if ee.get("theoretical_max_fps"):
            print("  Theoretical max FPS     : {:>8.2f}  (1000 / avg_inference_ms)".format(
                ee["theoretical_max_fps"]))

        print("\n" + SEP + "\n")
