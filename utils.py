import sys
import time
from config import STEPS


def progress_bar(current, total, task="", bar_len=35, start_time=None):
    pct = current / total
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    elapsed = time.time() - start_time if start_time else 0
    eta = (elapsed / pct - elapsed) if pct > 0.01 else 0
    eta_str = f"  ETA {eta:.0f}s" if 0 < pct < 1 else "        "
    sys.stdout.write(f"\r  [{bar}] {pct*100:5.1f}%  {task}{eta_str}")
    sys.stdout.flush()
    if current >= total:
        print()


def step_header(n, name):
    print(f"\n[{n}/{len(STEPS)}] {name}")