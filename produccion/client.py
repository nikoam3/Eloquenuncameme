# client.py
from binance.spot import Spot as Client
from config import BINANCE_KEY, BINANCE_SECRET

_binance_client = None

def get_client() -> Client:
    """
    Retorna la única instancia del cliente Binance (Singleton).
    Compartida por market.py y orders.py.
    """
    global _binance_client
    if _binance_client is None:
        _binance_client = Client(BINANCE_KEY, BINANCE_SECRET)
    return _binance_client