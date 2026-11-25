"""
simulate_attack.py
-------------------
CLI attack launcher — run this in a second terminal during the demo
to trigger a named attack scenario that the dashboard detects live.

Usage:
    .venv\\Scripts\\python.exe simulate_attack.py --scenario DDoS --n 20
    .venv\\Scripts\\python.exe simulate_attack.py --scenario DoS  --n 15
    .venv\\Scripts\\python.exe simulate_attack.py --scenario ALL  --n 30
    .venv\\Scripts\\python.exe simulate_attack.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from capture.simulator import FlowSimulator, ATTACK_SCENARIO_LABELS
from backend import alert_store

SHORTHAND = {
    "DDoS":   "TCP_IP-DDoS-UDP2_train",
    "DoS":    "TCP_IP-DoS-UDP_test",
    "MIXED":  "__MIXED__",
    "BENIGN": "__BENIGN__",
    "ALL":    "__ALL__",
}

RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"


def banner(scenario_key: str):
    label = ATTACK_SCENARIO_LABELS.get(scenario_key, scenario_key)
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{RED}{BOLD}  ⚡ IoMT SHIELD — ATTACK SIMULATOR{RESET}")
    print(f"  Scenario : {YELLOW}{label}{RESET}")
    print(f"  Time     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}{'='*60}{RESET}\n")


async def run(scenario_key: str, n: int, delay: float):
    banner(scenario_key)
    sim      = FlowSimulator(scenario=scenario_key, delay=delay)
    attacks  = 0
    benign   = 0

    async for ev in sim.stream(n=n):
        is_attack = ev["prediction"] == "Attack"
        if is_attack:
            attacks += 1
            alert_store.save_alert(ev)
            tag   = f"{RED}🚨 ATTACK {RESET}"
            extra = f"[{YELLOW}{ev['severity']}{RESET}] {ev['attack_detail']}"
        else:
            benign += 1
            tag   = f"{GREEN}✅ BENIGN {RESET}"
            extra = ""

        print(
            f"[{ev['timestamp']}] {tag} | "
            f"{ev['device_name']:<35s} | "
            f"{ev['ward']:<12s} | "
            f"Conf: {ev['confidence']:5.1f}% | "
            f"{extra}"
        )

    print(f"\n{BOLD}--- Stream complete ---{RESET}")
    print(f"Attacks: {RED}{attacks}{RESET}  |  Benign: {GREEN}{benign}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="IoMT Shield attack simulator CLI")
    parser.add_argument("--scenario", "-s", default="MIXED",
                        choices=list(SHORTHAND.keys()),
                        help="Attack scenario to simulate")
    parser.add_argument("--n",  type=int,   default=30,  help="Number of events")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between events")
    parser.add_argument("--list", "-l", action="store_true", help="List available scenarios")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for short, key in SHORTHAND.items():
            print(f"  {BOLD}{short:<8}{RESET} → {ATTACK_SCENARIO_LABELS.get(key, key)}")
        return

    key = SHORTHAND[args.scenario]
    asyncio.run(run(key, args.n, args.delay))


if __name__ == "__main__":
    main()

