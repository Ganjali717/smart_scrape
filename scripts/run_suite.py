"""
Run several experiment configs one after another (e.g. overnight); a failing config is
logged and the suite continues. Each run's console output goes to results/<name>.log.

    python scripts/run_suite.py configs/experiments/swde_book_semfeat.yaml \
        configs/experiments/swde_auto_semfeat.yaml configs/experiments/swde_job_semfeat.yaml
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from smartscrape.experiment import run  # noqa: E402


def main(paths):
    os.makedirs("results", exist_ok=True)
    for path in paths:
        name = yaml.safe_load(open(path, encoding="utf-8"))["name"]
        log_path = os.path.join("results", f"{name}.log")
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as lf:
            def log(*a):
                msg = " ".join(str(x) for x in a)
                print(msg, flush=True)
                lf.write(msg + "\n")
                lf.flush()
            log(f"##### {path}  started {time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                run(path, log=log)
                log(f"##### {path}  finished in {(time.time() - t0) / 60:.1f} min")
            except Exception:
                log(f"##### {path}  FAILED after {(time.time() - t0) / 60:.1f} min")
                log(traceback.format_exc())


if __name__ == "__main__":
    main(sys.argv[1:])
