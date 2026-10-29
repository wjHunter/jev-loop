import pytest
import requests

from jevloop.assets import resolve_symbol
from jevloop.execution.alpaca import (
    PAPER_TRADING_BASE_URL,
    AlpacaConfigError,
    AlpacaPaperClient,
    MarketClosedError,
    assert_paper_url,
)

BTC_SPEC = resolve_symbol("BTC/USD")


def test_paper_url_passes():
    assert_paper_url(PAPER_TRADING_BASE_URL)
    assert_paper_url(PAPER_TRADING_BASE_URL + "/")  # trailing slash tolerated


def test_live_url_is_refused():
    with pytest.raises(AlpacaConfigError):
        assert_paper_url("https://api.alpaca.markets")


def test_client_construction_refuses_live_base_url():
    with pytest.raises(AlpacaConfigError):
        AlpacaPaperClient(
            api_key="x",
            secret_key="y",
            spec=BTC_SPEC,
            base_url="https://api.alpaca.markets",
        )


def test_client_construction_accepts_paper_base_url():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    assert client.base_url == PAPER_TRADING_BASE_URL


def test_client_construction_works_for_equity_spec():
    equity_spec = resolve_symbol("AAPL")
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=equity_spec)
    assert client.symbol == "AAPL"
    assert client.spec.asset_class == "us_equity"


# -- closed-market guard: never place an equity order while the market is shut --


def test_limit_order_refused_when_equity_market_closed():
    equity_spec = resolve_symbol("AAPL")
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=equity_spec)
    client.is_market_open = lambda: False  # no network: force the closed branch
    with pytest.raises(MarketClosedError):
        client.submit_limit_order("buy", 1.0, 100.0)


def test_market_order_refused_when_equity_market_closed():
    equity_spec = resolve_symbol("AAPL")
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=equity_spec)
    client.is_market_open = lambda: False
    with pytest.raises(MarketClosedError):
        client.submit_market_order("sell", 1.0)


def test_crypto_never_checks_market_hours():
    # is_24_7 short-circuits the check entirely; is_market_open should not
    # even be consulted for a crypto spec. No network: _request is stubbed.
    crypto_client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)

    def _boom():
        raise AssertionError("is_market_open should never be called for a 24/7 asset")

    crypto_client.is_market_open = _boom
    crypto_client._request = lambda method, url, **kwargs: {"id": "fake-order"}

    # Use an order that meets the minimum notional ($10)
    result = crypto_client.submit_limit_order("buy", 0.001, 100000.0)  # $100
    assert result == {"id": "fake-order"}


# -- fractional equity orders must use time_in_force=day --


def test_fractional_equity_limit_order_uses_day_tif():
    """Fractional equity orders must use time_in_force=day, not gtc."""
    equity_spec = resolve_symbol("SPY")
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=equity_spec)
    client.is_market_open = lambda: True
    
    seen_bodies = []
    client._request = lambda method, url, json=None, **kwargs: (seen_bodies.append(json), {"id": "test-order"})[1]
    
    # Fractional order (0.5 shares)
    client.submit_limit_order("buy", 0.5, 450.0)
    assert seen_bodies[0]["time_in_force"] == "day"
    
    # Another fractional order (1.234 shares)
    client.submit_limit_order("sell", 1.234, 450.0)
    assert seen_bodies[1]["time_in_force"] == "day"


def test_whole_share_equity_limit_order_preserves_tif():
    """Whole-share equity orders can use the specified tif (default gtc)."""
    equity_spec = resolve_symbol("SPY")
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=equity_spec)
    client.is_market_open = lambda: True
    
    seen_bodies = []
    client._request = lambda method, url, json=None, **kwargs: (seen_bodies.append(json), {"id": "test-order"})[1]
    
    # Whole share order with default tif
    client.submit_limit_order("buy", 10.0, 450.0)
    assert seen_bodies[0]["time_in_force"] == "gtc"
    
    # Whole share order with explicit tif
    client.submit_limit_order("buy", 5.0, 450.0, tif="ioc")
    assert seen_bodies[1]["time_in_force"] == "ioc"


def test_crypto_limit_order_uses_gtc():
    """Crypto orders continue to use gtc as before."""
    crypto_client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    seen_bodies = []
    crypto_client._request = lambda method, url, json=None, **kwargs: (seen_bodies.append(json), {"id": "test-order"})[1]
    
    # Fractional crypto order (0.0012 BTC)
    crypto_client.submit_limit_order("buy", 0.0012, 95000.0)
    assert seen_bodies[0]["time_in_force"] == "gtc"


# -- retry logic for transient failures --


def test_request_retries_on_timeout():
    """Timeouts are retried with exponential backoff."""
    import time
    from unittest.mock import Mock
    
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    # First two calls timeout, third succeeds
    attempts = [0]
    def mock_request(method, url, timeout=10, **kwargs):
        attempts[0] += 1
        if attempts[0] <= 2:
            raise requests.exceptions.Timeout("Timed out")
        # Success on third attempt
        resp = Mock()
        resp.status_code = 200
        resp.text = '{"id": "success"}'
        resp.json = lambda: {"id": "success"}
        return resp
    
    client._session.request = mock_request
    client._limiter.wait = lambda: None  # skip rate limiting for test
    
    result = client._request("GET", "https://example.com/test")
    assert result == {"id": "success"}
    assert attempts[0] == 3


def test_request_retries_on_5xx():
    """5xx errors are retried with exponential backoff."""
    from unittest.mock import Mock
    
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    attempts = [0]
    def mock_request(method, url, timeout=10, **kwargs):
        attempts[0] += 1
        resp = Mock()
        if attempts[0] <= 2:
            resp.status_code = 503
            resp.text = "Service Unavailable"
        else:
            resp.status_code = 200
            resp.text = '{"id": "success"}'
            resp.json = lambda: {"id": "success"}
        return resp
    
    client._session.request = mock_request
    client._limiter.wait = lambda: None
    
    result = client._request("GET", "https://example.com/test")
    assert result == {"id": "success"}
    assert attempts[0] == 3


def test_request_retries_on_429():
    """429 (rate limit) errors are retried with exponential backoff."""
    from unittest.mock import Mock
    
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    attempts = [0]
    def mock_request(method, url, timeout=10, **kwargs):
        attempts[0] += 1
        resp = Mock()
        if attempts[0] == 1:
            resp.status_code = 429
            resp.text = "Too Many Requests"
        else:
            resp.status_code = 200
            resp.text = '{"id": "success"}'
            resp.json = lambda: {"id": "success"}
        return resp
    
    client._session.request = mock_request
    client._limiter.wait = lambda: None
    
    result = client._request("GET", "https://example.com/test")
    assert result == {"id": "success"}
    assert attempts[0] == 2


def test_request_fails_after_max_retries():
    """After max retries, the error is raised."""
    from unittest.mock import Mock
    
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    def mock_request(method, url, timeout=10, **kwargs):
        raise requests.exceptions.Timeout("Timed out")
    
    client._session.request = mock_request
    client._limiter.wait = lambda: None
    
    from jevloop.execution.alpaca import AlpacaAPIError
    with pytest.raises(AlpacaAPIError) as exc_info:
        client._request("GET", "https://example.com/test")
    assert "after 4 attempts" in str(exc_info.value)


def test_request_does_not_retry_permanent_errors():
    """4xx errors (except 429) are not retried."""
    from unittest.mock import Mock
    
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    attempts = [0]
    def mock_request(method, url, timeout=10, **kwargs):
        attempts[0] += 1
        resp = Mock()
        resp.status_code = 422
        resp.text = "fractional orders must be DAY orders"
        return resp
    
    client._session.request = mock_request
    client._limiter.wait = lambda: None
    
    from jevloop.execution.alpaca import AlpacaAPIError
    with pytest.raises(AlpacaAPIError) as exc_info:
        client._request("GET", "https://example.com/test")
    assert exc_info.value.status_code == 422
    assert attempts[0] == 1  # No retries for 4xx errors


# -- minimum notional enforcement --


def test_limit_order_below_min_notional_is_rejected():
    """Orders below min_notional_usd are rejected before submission."""
    crypto_client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    # Try to submit a $5 order (below $10 minimum for crypto)
    from jevloop.execution.alpaca import AlpacaAPIError
    with pytest.raises(AlpacaAPIError) as exc_info:
        crypto_client.submit_limit_order("buy", 0.00005, 100000.0)  # $5
    assert "below minimum" in str(exc_info.value)
    assert "$5.00" in str(exc_info.value)


def test_limit_order_above_min_notional_is_accepted():
    """Orders at or above min_notional_usd are accepted."""
    crypto_client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    crypto_client._request = lambda method, url, **kwargs: {"id": "test-order"}
    
    # $10 exactly (at minimum)
    result = crypto_client.submit_limit_order("buy", 0.0001, 100000.0)
    assert result == {"id": "test-order"}
    
    # $20 (well above minimum)
    result = crypto_client.submit_limit_order("buy", 0.0002, 100000.0)
    assert result == {"id": "test-order"}


def test_market_order_below_min_notional_is_rejected():
    """Market orders below min_notional_usd are rejected before submission."""
    crypto_client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    # Mock _request to return quote data
    def mock_request(method, url, **kwargs):
        if "latest/quotes" in url:
            return {"quotes": {"BTC/USD": {"ap": 100000.0, "bp": 99900.0}}}
        return {"id": "test-order"}
    
    crypto_client._request = mock_request
    crypto_client._limiter.wait = lambda: None
    
    # Try to submit a $5 buy (below $10 minimum)
    from jevloop.execution.alpaca import AlpacaAPIError
    with pytest.raises(AlpacaAPIError) as exc_info:
        crypto_client.submit_market_order("buy", 0.00005)  # ~$5 at $100k
    assert "below minimum" in str(exc_info.value)


def test_market_order_above_min_notional_is_accepted():
    """Market orders at or above min_notional_usd are accepted."""
    crypto_client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC_SPEC)
    
    def mock_request(method, url, **kwargs):
        if "latest/quotes" in url:
            return {"quotes": {"BTC/USD": {"ap": 100000.0, "bp": 99900.0}}}
        return {"id": "test-order"}
    
    crypto_client._request = mock_request
    crypto_client._limiter.wait = lambda: None
    
    # $10+ buy
    result = crypto_client.submit_market_order("buy", 0.00011)  # ~$11 at $100k ask
    assert result == {"id": "test-order"}
    
    # $10+ sell
    result = crypto_client.submit_market_order("sell", 0.00011)  # ~$10.99 at $99.9k bid
    assert result == {"id": "test-order"}


def test_equity_min_notional_check():
    """Equity orders also enforce min_notional ($1 for equities)."""
    equity_spec = resolve_symbol("SPY")
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=equity_spec)
    client.is_market_open = lambda: True
    
    # Try to submit a $0.50 order (below $1 minimum for equity)
    from jevloop.execution.alpaca import AlpacaAPIError
    with pytest.raises(AlpacaAPIError) as exc_info:
        client.submit_limit_order("buy", 0.001, 450.0)  # $0.45
    assert "below minimum" in str(exc_info.value)
    
    # $1.00+ should work
    client._request = lambda method, url, **kwargs: {"id": "test-order"}
    result = client.submit_limit_order("buy", 0.003, 450.0)  # $1.35
    assert result == {"id": "test-order"}
