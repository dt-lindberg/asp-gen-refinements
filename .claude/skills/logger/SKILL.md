---
name: logger
description: Documents the logging setup in this project. Use when adding a new module that needs logging, debugging missing log output, or changing which modules are allowed to log.
---

# Logger Setup

Centralized in `logger.py`. All modules use the same two functions:

```python
from logger import setup_logging, get_logger

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)
```

`setup_logging` is safe to call multiple times per process — each module-level call is a no-op after the first unless `force=True` is passed (due to `logging.basicConfig` behavior).

## How it works

1. **Handlers** — writes to both `stdout` and a timestamped file under `$LOG_DIR/<YYYYMMDD>/log_<timestamp>.log`.
2. **Root logger** — set to `WARNING`, which silences all loggers by default.
3. **Allowlist** — specific module loggers are explicitly elevated to the configured level:

```python
# logger.py:59
for name in ("__main__", "pipeline", "refinement_loop"):
    logging.getLogger(name).setLevel(log_level)
```

Logger names come from `__name__`, which equals the module filename without `.py`.

## Adding a new module

Add its name to the allowlist tuple in `logger.py:59`:

```python
for name in ("__main__", "pipeline", "refinement_loop", "your_module"):
    logging.getLogger(name).setLevel(log_level)
```

## Log level

Controlled by the `LOG_LEVEL` env var (default: `"debug"`). Valid values: `debug`, `info`, `warning`, `error`, `critical`.

## Log format

```
%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d > %(message)s
```
