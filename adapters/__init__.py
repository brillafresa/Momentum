"""External dependency adapters (market data, universe, persistence).

Market data should be injected via a port/protocol so core scoring can run
offline against fixtures.

- ``market_data``: ``MarketDataPort`` / ``YFinanceAdapter`` / ``FixtureAdapter``
- ``price_cache``: disk last-bar cache shared by batch and UI (runtime ``cache/``)
- ``ui_data_bundle``: DetailViewAtom + session fingerprints (no network)

Concrete adapters land here as migration proceeds. Production entrypoints
(``app.py``, ``run_scan_batch.py``) must not import ``tests`` or ``harness``.
"""
