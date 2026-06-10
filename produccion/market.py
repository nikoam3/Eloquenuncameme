import requests
import pandas as pd
import logging
from binance.spot import Spot as Client
from config import BINANCE_KEY, BINANCE_SECRET, PAR

_binance_client = None

def get_client():
    """
    Retorna la instancia del cliente de Binance.
    Aplica el patrón Singleton para evitar crear múltiples conexiones HTTP.
    """
    global _binance_client
    if _binance_client is None:
        # Solo se inicializa la primera vez que se llama
        _binance_client = Client(BINANCE_KEY, BINANCE_SECRET)
    return _binance_client

def get_decimales(par: str):
    """Obtiene la cantidad de decimales para precio y cantidad del par"""
    spot_client = Client()
    info = spot_client.exchange_info(symbol=par)['symbols'][0]['filters']
    tickSize = float(info[0]['tickSize'])
    stepSize = float(info[1]['stepSize'])

    def busca_decimal(step):
        mult = step
        decimal = 0
        while mult != 1:
            decimal += 1
            mult *= 10
        return decimal

    return busca_decimal(tickSize), busca_decimal(stepSize)

def get_precios(par: str, intervalo: str) -> pd.DataFrame:
    """Descarga las últimas 1000 velas del par e intervalo indicados"""
    try:
        url = 'https://api.binance.com/api/v3/klines'
        p = {'symbol': par, 'interval': intervalo, 'limit': 1000}
        r = requests.get(url, params=p)
        r.raise_for_status()  # Lanza un error si la respuesta no es 200
        js = r.json()
    except requests.RequestException as e:
        logging.error(f"Error al obtener precios de Binance: {e}")
        raise

    col = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_q',
           'cTime', 'trades', 'takerBase', 'takerQuote', 'Ignore']
    df = pd.DataFrame(js, columns=col)
    df = df.apply(pd.to_numeric)
    df.index = pd.to_datetime(df.Time, unit='ms')
    df = df.drop(['Time', 'Volume_q', 'cTime', 'trades', 
                  'takerBase', 'takerQuote', 'Ignore'], axis=1)
    return df

def get_order_book(par: str, decimal_price: int) -> pd.DataFrame:
    
    """Obtiene las mejores ofertas de compra y venta"""
    try:
        url = 'https://api.binance.com/api/v3/depth'
        p = {'symbol': par, 'limit': 5}
        r = requests.get(url, params=p)
        r.raise_for_status()  # Lanza un error si la respuesta no es 200
        js = r.json()
    except requests.RequestException as e:
        logging.error(f"Error al obtener el order book de Binance: {e}")
        raise

    bids = pd.DataFrame(js['bids'])
    asks = pd.DataFrame(js['asks'])
    df = pd.concat([bids[1], bids[0], asks[0], asks[1]], axis=1)
    df.columns = ['bid_quant', 'bid_price', 'ask_price', 'ask_quant']
    df = df.apply(pd.to_numeric).round(decimal_price)
    return df

def get_saldo(asset: str) -> float:
    """
    Retorna el saldo disponible de un asset (ej: 'USDT' o 'GALA').
    Maneja de forma segura el caso en que el asset no exista en la cuenta
    o si hay problemas de conexión con la API.
    """
    try:
        client = get_client()
        balances = client.account()['balances']
        
        # Le damos a next() un valor por defecto (None) si no encuentra el asset
        asset_data = next((x for x in balances if x["asset"] == asset), None)
        
        if asset_data:
            return float(asset_data['free'])
        else:
            # Si no existe en la wallet, asumimos con seguridad que el saldo es cero
            logging.info(f"El activo {asset} no figura en los balances. Retornando saldo 0.0")
            return 0.0
            
    except Exception as e:
        # En sistemas financieros, si la consulta falla (ej. error de red), 
        # devolvemos 0.0 para evitar que el bot intente operar con dinero fantasma.
        logging.error(f"🛑 Error crítico consultando saldo de {asset}: {e}")
        return 0.0