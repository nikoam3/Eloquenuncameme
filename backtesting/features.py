import pandas as pd
import numpy as np
import pandas_ta as ta
from database import cargar_velas

# ============================================================
# CONFIGURACIÓN
# ============================================================
PAR       = "ETHUSDT"
INT_15M   = "15m"
INT_1H    = "1h"


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
def cargar_y_calcular(par: str, intervalo: str) -> pd.DataFrame:
    """
    Carga velas desde la BD y calcula todos los indicadores técnicos.
    """
    df = cargar_velas(par, intervalo)

    # Tendencia
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['EMA50']  = ta.ema(df['Close'], length=50)

    # Momentum
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ROC'] = ta.roc(df['Close'], length=10)

    # Volatilidad
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Bandas de Bollinger (cálculo manual para evitar diferencias
    # en nombres de columnas devueltas por pandas_ta)
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_mid = ma20
    df['BB_ancho'] = (bb_upper - bb_lower) / bb_mid.replace({0: np.nan})

    # Fuerza de tendencia
    adx       = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']

    # Volumen relativo
    df['Vol_media'] = ta.sma(df['Volume'], length=20)
    df['Vol_rel']   = df['Volume'] / df['Vol_media']

    # MACD solo histograma
    df['MACD_hist'] = ta.macd(df['Close'], fast=12, slow=26, signal=6)['MACDh_12_26_6']

    df = df.dropna()
    return df


# ============================================================
# FEATURES MULTI-TIMEFRAME
# ============================================================
def construir_features(par: str = PAR) -> pd.DataFrame:
    """
    Construye el DataFrame de features combinando 15m y 1h.
    Cada fila representa un momento en el tiempo con toda
    la información disponible HASTA ese momento (sin leakage).
    """
    print(f"Cargando datos {par}...")
    df_15m = cargar_y_calcular(par, INT_15M)
    df_1h  = cargar_y_calcular(par, INT_1H)

    print(f"  15m: {len(df_15m):,} velas")
    print(f"  1h:  {len(df_1h):,} velas")

    # ── Features del timeframe 15m ──────────────────────────

    features = pd.DataFrame(index=df_15m.index)

    # Tendencia: distancia relativa al EMA200
    features['close_vs_ema200'] = (
        (df_15m['Close'] - df_15m['EMA200']) / df_15m['EMA200']
    )

    # Tendencia: relación EMA50 vs EMA200
    features['ema50_vs_ema200'] = (
        (df_15m['EMA50'] - df_15m['EMA200']) / df_15m['EMA200']
    )

    # Momentum: RSI y su pendiente
    features['rsi'] = df_15m['RSI']
    #features['rsi_pendiente'] = df_15m['RSI'].diff(3)  # cambio en 3 velas

    # Momentum: Rate of Change
    #features['roc'] = df_15m['ROC']

    # Volatilidad: ATR normalizado por precio
    features['atr_relativo'] = df_15m['ATR'] / df_15m['Close']
    
    # atr_tendencia  → ¿el ATR está subiendo? (volatilidad creciente)
    features['atr_tendencia'] = df_15m['ATR'].diff(4) / df_15m['ATR']

    # Volatilidad: ancho de Bandas de Bollinger
    features['bb_ancho'] = df_15m['BB_ancho']

    # Fuerza de tendencia
    features['adx'] = df_15m['ADX']

    # Volumen relativo
    #features['vol_relativo'] = df_15m['Vol_rel'].clip(0, 5)  # cap en 5x

    #rsi_pendiente_vs_precio: compara cuanto cambio el precio (%) vs cuanto cambio el rsi
    cambio_precio_pct = (
        df_15m['Close'] / df_15m['Close'].shift(20) - 1
    ) * 100
    cambio_rsi = df_15m['RSI'] - df_15m['RSI'].shift(20)

    # Si el precio sube mucho y el RSI sube poco (o baja),
    # el ratio se vuelve alto/negativo => señal de divergencia
    # Usamos un epsilon para evitar división por cero
    features['rsi_pendiente_vs_precio'] = (
        cambio_precio_pct - cambio_rsi
    )

    # Variante 1: histograma normalizado por precio (para que sea
    # comparable entre distintos niveles de precio de ETH en el tiempo)
    features['macd_histograma'] = df_15m['MACD_hist'] / df_15m['Close'] * 1000


    # Contexto temporal
    features['hora']       = df_15m.index.hour
    features['dia_semana'] = df_15m.index.dayofweek

    # ── Features del timeframe 1h (contexto macro) ──────────
    # Usamos .reindex + ffill para alinear temporalmente
    # ffill = forward fill: cada vela de 15m toma el valor
    # de la última vela de 1h completada → sin leakage

    df_1h_reindexed = df_1h.reindex(
        df_15m.index, method='ffill'
    )

    features['tendencia_1h'] = (
        (df_1h_reindexed['Close'] > df_1h_reindexed['EMA200'])
        .astype(int)
    )

    features['rsi_1h'] = df_1h_reindexed['RSI']
    features['adx_1h'] = df_1h_reindexed['ADX']
    features['roc_1h'] = df_1h_reindexed['ROC']

    features = features.dropna()

    print(f"\n✅ Features construidas: {len(features):,} filas x "
          f"{len(features.columns)} columnas")
    print(f"   Período: {features.index[0]} → {features.index[-1]}")

    return features, df_15m


# ============================================================
# CREAR EL TARGET
# ============================================================
def crear_target(df_15m: pd.DataFrame,
                 stop_loss:    float = 0.025,
                 take_profit:  float = 0.010,
                 max_velas:    int   = 96) -> pd.Series:
    """
    Para cada vela, simula una operación de compra y determina
    si hubiera sido ganadora (1) o perdedora (0).

    Parámetros:
        stop_loss:   porcentaje de pérdida máxima (0.025 = 2.5%)
        take_profit: porcentaje de ganancia objetivo (0.010 = 1%)
        max_velas:   máximo de velas a esperar (96 x 15m = 24 horas)

    Lógica:
        Compramos al Open de la vela siguiente.
        Miramos hacia adelante vela por vela.
        Si el precio toca take_profit → ganadora (1)
        Si el precio toca stop_loss   → perdedora (0)
        Si pasan max_velas sin tocar ninguno → perdedora (0)
    """
    closes  = df_15m['Close'].values
    highs   = df_15m['High'].values
    lows    = df_15m['Low'].values
    opens   = df_15m['Open'].values
    n       = len(df_15m)
    target  = np.full(n, np.nan)

    for i in range(n - max_velas):
        precio_entrada = opens[i + 1]  # compramos al open siguiente
        precio_tp      = precio_entrada * (1 + take_profit)
        precio_sl      = precio_entrada * (1 - stop_loss)

        resultado = 0  # perdedora por defecto

        for j in range(i + 1, min(i + 1 + max_velas, n)):
            if highs[j] >= precio_tp:
                resultado = 1  # tocó take profit → ganadora
                break
            if lows[j] <= precio_sl:
                resultado = 0  # tocó stop loss → perdedora
                break

        target[i] = resultado

    return pd.Series(target, index=df_15m.index, name='target')


# ============================================================
# ANÁLISIS DEL TARGET
# ============================================================
def analizar_target(target: pd.Series):
    """
    Muestra estadísticas del target para verificar que
    el balance de clases es razonable.
    """
    target_limpio = target.dropna()
    ganadoras     = int(target_limpio.sum())
    perdedoras    = int(len(target_limpio) - ganadoras)
    total         = len(target_limpio)

    print("\n" + "="*45)
    print("   ANÁLISIS DEL TARGET")
    print("="*45)
    print(f"  Total muestras:  {total:,}")
    print(f"  Ganadoras (1):   {ganadoras:,}  ({ganadoras/total*100:.1f}%)")
    print(f"  Perdedoras (0):  {perdedoras:,}  ({perdedoras/total*100:.1f}%)")

    ratio = ganadoras / perdedoras if perdedoras > 0 else float('inf')
    print(f"  Ratio G/P:       {ratio:.2f}")

    if ratio < 0.5 or ratio > 2.0:
        print("\n  ⚠️  Clases muy desbalanceadas.")
        print("     Considerar técnicas de balanceo (SMOTE o class_weight).")
    else:
        print("\n  ✅ Balance de clases razonable.")
    print("="*45)


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    # 1. Construimos features
    features, df_15m = construir_features(PAR)

    # 2. Creamos el target
    print("\nCreando target (puede tardar un momento)...")
    target = crear_target(df_15m, 
                          stop_loss=0.025,
                          take_profit=0.010,
                          max_velas=96)

    # 3. Analizamos el target
    analizar_target(target)

    # 4. Alineamos features y target
    df_ml = features.join(target).dropna()

    print(f"\n  Dataset final: {len(df_ml):,} filas x "
          f"{len(df_ml.columns)} columnas")
    print(f"  Features: {list(features.columns)}")

    # 5. Guardamos para usar en el siguiente paso
    try:
        df_ml.to_parquet('dataset_ml.parquet')
        print("\n✅ Dataset guardado en dataset_ml.parquet")
        print("DataFrame final", df_ml)
    except ImportError:
        print("\n⚠️  No se encontró un motor Parquet válido.")
        print("   Guardando dataset como CSV en su lugar...")
        df_ml.to_csv('dataset_ml.csv', index=True)
        print("\n✅ Dataset guardado en dataset_ml.csv")