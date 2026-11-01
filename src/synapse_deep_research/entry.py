"""
Console-script entry point.

A thin wrapper so the carry-over from the former `deep-research` install can run
before main.py (and through it config.py) is imported: config reads the global
.env, the saved settings and the data dir at import time, and those must already
be in place on the first launch under the new name.
"""
from __future__ import annotations

import logging


def run() -> None:
    try:
        from . import legacy_install

        carried = legacy_install.carry_over()
        if carried:
            from .ui import note, ok

            ok(f"Carried over from your old `{legacy_install.OLD_NAME}` install: {legacy_install.describe(carried)}.")
            where = [str(d) for d in carried.old_dirs]
            if carried.keys:
                where.append(f"the `{legacy_install.OLD_NAME}` keyring entries")
            note(f"The old copies were left in place ({' and '.join(where)}); delete them once you no longer need them.")
    except Exception:  # noqa: BLE001 — a failed carry-over must never keep the CLI from starting
        logging.getLogger(__name__).debug("carry-over from the old name failed", exc_info=True)

    from .main import main

    main()
