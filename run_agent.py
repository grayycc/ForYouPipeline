#!/usr/bin/env python3
"""Entrypoint for the KuaiRand autonomous ML agent.

    python3 run_agent.py --run_id run1
    python3 run_agent.py --run_id smoke --max_iterations 2
"""
import argparse
import datetime
import sys

from agent import config
from agent.loop import Agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', default=datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    ap.add_argument('--max_iterations', type=int, default=config.MAX_ITERATIONS)
    ap.add_argument('--seed', type=int, default=0, help='seed for the agent search policy')
    a = ap.parse_args()

    print(f'run_id={a.run_id}  region={config.AWS_REGION}  '
          f'models={config.STRONG_MODEL}/{config.FAST_MODEL}  cap={a.max_iterations}')
    agent = Agent(a.run_id, seed=a.seed)
    try:
        agent.run(max_iterations=a.max_iterations)
    except KeyboardInterrupt:
        print('\n[interrupted] writing summary before exit')
        agent.write_summary()
        sys.exit(130)


if __name__ == '__main__':
    main()
