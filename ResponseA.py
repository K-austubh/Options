# Below is a single‐file Python module named `option_mc.py`. It provides:

# • A Bloomberg data–provider stub (using the official `blpapi` library)  
# • Monte Carlo pricers for European and barrier options  
# • Proper logging, type hints, and docstrings  
# • Built-in unit tests using the standard `unittest` framework  

# Save this as `option_mc.py`. You will need to install dependencies (`numpy`, `blpapi`) and configure the Bloomberg API environment per your IT/regulatory rules.

"""
option_mc.py

A single-file module for Monte Carlo pricing of European and barrier options,
with integration hooks for Bloomberg real-time market data.
"""

import logging
import math
import datetime
from typing import Optional, Literal
import numpy as np

# Attempt to import Bloomberg API; application must install/configure blpapi
try:
    import blpapi
except ImportError:
    blpapi = None

# Configure module‐level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_handler)


class BloombergDataProvider:
    """
    Market data provider using Bloomberg API (blpapi).
    Requires a running Bloomberg terminal or server and correct environment.
    """

    def __init__(self, host: str = "localhost", port: int = 8194) -> None:
        if blpapi is None:
            raise ImportError("blpapi is not installed; install it per Bloomberg documentation.")
        self.session = blpapi.Session(
            blpapi.SessionOptions().setServerHost(host).setServerPort(port)
        )
        if not self.session.start():
            raise ConnectionError("Failed to start Bloomberg session")
        if not self.session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdata service")
        self.ref_data_service = self.session.getService("//blp/refdata")
        logger.info("BloombergDataProvider initialized (host=%s port=%d)", host, port)

    def get_spot_price(self, ticker: str, field: str = "PX_LAST") -> float:
        """
        Fetch the latest spot price for a given ticker.
        """
        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue(field)
        self.session.sendRequest(request)

        while True:
            ev = self.session.nextEvent(500)
            for msg in ev:
                if msg.hasElement("securityData"):
                    sd = msg.getElement("securityData").getValueAsElement(0)
                    fieldData = sd.getElement("fieldData")
                    price = fieldData.getElementAsFloat(field)
                    logger.debug("Fetched %s for %s: %f", field, ticker, price)
                    return price
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
        raise RuntimeError(f"No data returned for {ticker} / {field}")


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    sims: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.
    Returns an array of shape (sims, steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)

    increments = rng.standard_normal(size=(sims, steps))
    increments = drift + vol * increments
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate((np.zeros((sims, 1)), log_paths), axis=1)
    paths = S0 * np.exp(log_paths)
    logger.debug("Simulated GBM paths shape: %s", paths.shape)
    return paths


def price_european_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    sims: int = 100_000,
    steps: int = 100,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Monte Carlo pricing of a European call or put.
    """
    logger.info("Pricing European %s: S0=%.2f K=%.2f T=%.2f r=%.2f sigma=%.2f sims=%d steps=%d",
                "Call" if is_call else "Put", S0, K, T, r, sigma, sims, steps)

    paths = simulate_gbm_paths(S0, r, sigma, T, steps, sims, rng)
    ST = paths[:, -1]
    if is_call:
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    price = math.exp(-r * T) * payoffs.mean()
    logger.info("European %s price: %.6f", "Call" if is_call else "Put", price)
    return price


BarrierType = Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"]


def price_barrier_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: BarrierType,
    is_call: bool = True,
    sims: int = 100_000,
    steps: int = 100,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Monte Carlo pricing of a barrier option.
    barrier_type must be one of "up-and-out", "down-and-out", "up-and-in", "down-and-in".
    """
    logger.info("Pricing Barrier %s %s: S0=%.2f K=%.2f barrier=%.2f T=%.2f r=%.2f sigma=%.2f sims=%d steps=%d",
                barrier_type, "Call" if is_call else "Put", S0, K, barrier, T, r, sigma, sims, steps)

    paths = simulate_gbm_paths(S0, r, sigma, T, steps, sims, rng)
    ST = paths[:, -1]

    # Determine barrier breach
    if "up" in barrier_type:
        breached = np.any(paths >= barrier, axis=1)
    else:  # "down"
        breached = np.any(paths <= barrier, axis=1)

    # Determine option existence
    if barrier_type.endswith("out"):
        alive = ~breached
    else:  # "in"
        alive = breached

    # Payoff at expiry only if alive
    if is_call:
        payoff = np.where(alive, np.maximum(ST - K, 0.0), 0.0)
    else:
        payoff = np.where(alive, np.maximum(K - ST, 0.0), 0.0)

    price = math.exp(-r * T) * payoff.mean()
    logger.info("Barrier option price: %.6f", price)
    return price


# ------------------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------------------
import unittest


class TestOptionPricing(unittest.TestCase):
    def setUp(self) -> None:
        # Use a fixed RNG for reproducibility
        self.rng = np.random.default_rng(12345)
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_european_call(self):
        price = price_european_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                  is_call=True, sims=200_000, steps=200, rng=self.rng)
        # Black-Scholes reference ~10.45 for these parameters
        self.assertAlmostEqual(price, 10.45, places=1)

    def test_european_put(self):
        price = price_european_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                  is_call=False, sims=200_000, steps=200, rng=self.rng)
        # Put-call parity: put ≈ call - S0 + K*e^{-rT}
        call = price_european_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                 is_call=True, sims=200_000, steps=200, rng=self.rng)
        pvK = self.K * math.exp(-self.r * self.T)
        self.assertAlmostEqual(price, call - self.S0 + pvK, places=2)

    def test_barrier_up_and_out(self):
        price = price_barrier_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                 barrier=120.0, barrier_type="up-and-out",
                                 is_call=True, sims=200_000, steps=200, rng=self.rng)
        # Should be lower than plain vanilla
        vanilla = price_european_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                    is_call=True, sims=200_000, steps=200, rng=self.rng)
        self.assertLess(price, vanilla)

    def test_barrier_up_and_in(self):
        price_out = price_barrier_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                     barrier=120.0, barrier_type="up-and-out",
                                     is_call=True, sims=200_000, steps=200, rng=self.rng)
        price_in = price_barrier_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                    barrier=120.0, barrier_type="up-and-in",
                                    is_call=True, sims=200_000, steps=200, rng=self.rng)
        # In + Out ≈ vanilla
        vanilla = price_european_mc(self.S0, self.K, self.T, self.r, self.sigma,
                                    is_call=True, sims=200_000, steps=200, rng=self.rng)
        self.assertAlmostEqual(price_in + price_out, vanilla, places=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()


# How to use:

# 1. Install dependencies  
#    ```
#    pip install numpy blpapi
#    ```
# 2. Configure Bloomberg (refer to your internal IT/regulatory guide).  
# 3. In your real-time application, do something like:
#    ```python
#    from option_mc import BloombergDataProvider, price_european_mc

#    mdp = BloombergDataProvider()
#    spot = mdp.get_spot_price("AAPL US Equity")
#    vol = 0.25  # or fetch implied vol similarly
#    price = price_european_mc(spot, 150, 0.5, 0.01, vol, is_call=True)
#    ```
# 4. Run unit tests:  
#    ```
#    python option_mc.py
#    ```
  
# This single module should integrate neatly into your codebase and serve as a template for more advanced variance-reduction, Greeks, or multi-asset extensions.