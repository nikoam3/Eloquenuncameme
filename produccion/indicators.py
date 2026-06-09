import pandas_ta as ta
from market import get_precios
from config import PAR, INTERVALO

def calcular_indicadores(df):
    """
    Recibe un DataFrame con columnas OHLCV y agrega los indicadores técnicos.
    Retorna el mismo DataFrame con las columnas nuevas.
    """

    # Media móvil exponencial de 200 períodos (tendencia general)
    df['MM'] = ta.ema(df['Close'], length=200)

    # RSI de 14 períodos (sobrecompra/sobreventa)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # Suavizado del RSI con EMA de 2 períodos
    df['RSI_EMA'] = ta.ema(df['RSI'], length=2)

    # ADX de 14 períodos (fuerza de la tendencia)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']

    # Parabolic SAR (señal de reversión)
    sar = ta.psar(df['High'], df['Low'], df['Close'], af0=0.02, max_af=0.2)
    df['SAR'] = sar['PSARl_0.02_0.2'].fillna(sar['PSARs_0.02_0.2'])

    # Volumen promedio de 20 períodos
    df['Vol_Med'] = ta.sma(df['Volume'], length=20)

    return df


def get_datos(par: str = PAR, intervalo: str = INTERVALO):
    """
    Función principal: descarga precios y calcula todos los indicadores.
    Es lo que vas a llamar desde strategy.py y main.py
    """
    df = get_precios(par, intervalo)
    df = calcular_indicadores(df)
    return df