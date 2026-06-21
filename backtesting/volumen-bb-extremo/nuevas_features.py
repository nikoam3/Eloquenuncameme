import sys
import os
# Encuentra la carpeta principal 'backtesting' subiendo un nivel
ruta_principal = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ruta_principal)
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from features import construir_features, crear_target
from database import cargar_velas
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
PAR               = "ETHUSDT"
STOP_LOSS         = 0.025
TAKE_PROFIT       = 0.010
MAX_VELAS         = 96
N_VENTANAS        = 4
VENTANA_PENDIENTE = 20  # ya confirmada para RSI, punto de partida

# Baseline actualizado: incluye las DOS features ya confirmadas
# en producción (rsi_pendiente_vs_precio y macd_histograma)
FEATURES_BASE_ACTUALIZADO = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'rsi_pendiente_vs_precio',
    'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


# ============================================================
# CÁLCULO DE LAS TRES FEATURES NUEVAS
# ============================================================
def calcular_rsi_pendiente(df: pd.DataFrame,
                           ventana: int = VENTANA_PENDIENTE) -> pd.Series:
    """Ya confirmada en 7B.1 — la recalculamos para el baseline."""
    cambio_precio_pct = (
        df['Close'] / df['Close'].shift(ventana) - 1
    ) * 100
    cambio_rsi = df['RSI'] - df['RSI'].shift(ventana)
    return cambio_precio_pct - cambio_rsi


def calcular_macd_histograma(df: pd.DataFrame) -> pd.Series:
    """Ya confirmada en 7B.2 — parámetros (12,26,6), norm. por precio."""
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=6)
    hist = macd['MACDh_12_26_6']
    return hist / df['Close'] * 1000


def calcular_volumen_en_extremo(df: pd.DataFrame,
                                ventana: int = VENTANA_PENDIENTE) -> pd.Series:
    """
    Mide si los nuevos máximos/mínimos de precio vienen acompañados
    de volumen relativamente alto o bajo, comparado con el volumen
    promedio de la ventana.

    Valor alto positivo: nuevo máximo CON volumen fuerte (movimiento
                          "sano", muchas manos participando)
    Valor alto negativo: nuevo máximo CON volumen débil (posible
                          agotamiento, pocas manos lo sostienen)
    Cerca de cero: sin movimiento de extremo relevante en esta vela
    """
    vol_medio = df['Volume'].rolling(ventana).mean()
    vol_relativo = (df['Volume'] / vol_medio) - 1  # % sobre/bajo el promedio

    max_high_previo = df['High'].rolling(ventana).max().shift(1)
    min_low_previo  = df['Low'].rolling(ventana).min().shift(1)

    nuevo_max = (df['High'] > max_high_previo).astype(int)
    nuevo_min = (df['Low'] < min_low_previo).astype(int)

    # Si es nuevo máximo, multiplicamos por vol_relativo (signo +)
    # Si es nuevo mínimo, multiplicamos por vol_relativo pero invertido
    # (queremos que "mínimo con volumen fuerte" también dé señal,
    # pero conceptualmente distinta del máximo, así que las separamos
    # en dos términos y los sumamos con signos opuestos)
    señal_max = nuevo_max * vol_relativo
    señal_min = nuevo_min * vol_relativo * -1

    return señal_max + señal_min


def calcular_distancia_a_extremo(df: pd.DataFrame,
                                 ventana: int = 50) -> pd.Series:
    """
    Cuántas velas pasaron desde el último máximo o mínimo local
    (el que sea más reciente de los dos), normalizado por la
    ventana de búsqueda para que el valor quede entre 0 y 1.

    Valor cercano a 0: extremo muy reciente (movimiento "fresco")
    Valor cercano a 1: hace mucho que no hay extremo nuevo
                        (posible agotamiento/lateralización)
    """
    n = len(df)
    distancia = np.full(n, ventana, dtype=float)  # default: el máximo posible

    highs = df['High'].values
    lows  = df['Low'].values

    for i in range(ventana, n):
        ventana_high = highs[i-ventana:i+1]
        ventana_low  = lows[i-ventana:i+1]

        idx_max = np.argmax(ventana_high)  # posición del máximo en la ventana
        idx_min = np.argmin(ventana_low)

        # Distancia en velas desde el extremo más reciente hasta ahora
        velas_desde_max = ventana - idx_max
        velas_desde_min = ventana - idx_min

        distancia[i] = min(velas_desde_max, velas_desde_min)

    return pd.Series(distancia, index=df.index) / ventana


def calcular_bb_pendiente_vs_precio(df: pd.DataFrame,
                                    ventana: int = VENTANA_PENDIENTE,
                                    ventana_bb: int = 20) -> pd.Series:
    """
    Análoga a rsi_pendiente_vs_precio y macd_pendiente_vs_precio,
    pero usando la posición %B dentro de las Bandas de Bollinger
    en vez del RSI o el MACD.

    %B = (Close - banda_inferior) / (banda_superior - banda_inferior)
    """
    ma  = df['Close'].rolling(ventana_bb).mean()
    std = df['Close'].rolling(ventana_bb).std()
    banda_sup = ma + 2 * std
    banda_inf = ma - 2 * std

    pct_b = (df['Close'] - banda_inf) / (banda_sup - banda_inf)

    cambio_precio_pct = (
        df['Close'] / df['Close'].shift(ventana) - 1
    ) * 100
    cambio_pct_b = (pct_b - pct_b.shift(ventana)) * 100  # a escala %

    return cambio_precio_pct - cambio_pct_b


def walk_forward_auc_silencioso(X: pd.DataFrame, y: pd.Series) -> list:
    n        = len(X)
    tam_test = n // (N_VENTANAS + 1)
    aucs     = []

    for i in range(N_VENTANAS):
        fin_train = tam_test * (i + 1)
        fin_test  = tam_test * (i + 2)

        X_train = X.iloc[:fin_train]
        y_train = y.iloc[:fin_train]
        X_test  = X.iloc[fin_train:fin_test]
        y_test  = y.iloc[fin_train:fin_test]

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
        modelo.fit(X_tr_sc, y_train)
        probs = modelo.predict_proba(X_te_sc)[:, 1]

        aucs.append(roc_auc_score(y_test, probs))

    return aucs


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Evaluación de features pendientes (volumen, distancia, BB)\n")

    print("Cargando datos y calculando indicadores...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    df_15m = df_15m.dropna()

    print("Calculando features ya confirmadas (RSI pendiente, MACD)...")
    df_15m['rsi_pendiente_vs_precio'] = calcular_rsi_pendiente(df_15m)
    df_15m['macd_histograma']         = calcular_macd_histograma(df_15m)

    print("Calculando las TRES features nuevas a evaluar...")
    print("  - volumen_en_extremo...")
    df_15m['volumen_en_extremo'] = calcular_volumen_en_extremo(df_15m)
    print("  - distancia_a_extremo... (puede tardar unos segundos, tiene loop)")
    df_15m['distancia_a_extremo'] = calcular_distancia_a_extremo(df_15m)
    print("  - bb_pendiente_vs_precio...")
    df_15m['bb_pendiente_vs_precio'] = calcular_bb_pendiente_vs_precio(df_15m)

    print("\nConstruyendo features base (multi-timeframe)...")
    features_df, df_15m_completo = construir_features("ETHUSDT")
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, max_velas=MAX_VELAS
    )

    cols_nuevas = ['volumen_en_extremo', 'distancia_a_extremo',
                   'bb_pendiente_vs_precio']
    nuevas_alineadas = df_15m[cols_nuevas].reindex(features_df.index)

    df_ml = features_df.join(target).join(nuevas_alineadas).dropna()
    print(f"Dataset final: {len(df_ml):,} filas\n")

    y = df_ml['target']

    # ── Baseline actualizado (con RSI+MACD ya incluidas) ────
    print(f"{'='*60}")
    print("  Baseline (incluye rsi_pendiente + macd_histograma)")
    print(f"{'='*60}")
    X_base = df_ml[FEATURES_BASE_ACTUALIZADO]
    aucs_base = walk_forward_auc_silencioso(X_base, y)
    auc_base_prom = np.mean(aucs_base)
    print(f"  AUC promedio: {auc_base_prom:.4f}")
    print(f"  Por ventana: {[round(a, 4) for a in aucs_base]}")

    # ── Aporte individual de cada feature nueva ─────────────
    print(f"\n{'='*60}")
    print("  Aporte INDIVIDUAL de cada feature nueva")
    print(f"{'='*60}")
    print(f"  {'Feature agregada':<28} {'AUC':>8} {'vs base':>10} "
          f"{'std entre V':>12}")
    print(f"  {'─'*60}")

    features_nuevas = ['volumen_en_extremo', 'distancia_a_extremo',
                       'bb_pendiente_vs_precio']
    resultados = {}

    for feat in features_nuevas:
        cols = FEATURES_BASE_ACTUALIZADO + [feat]
        X_var = df_ml[cols]
        aucs_var = walk_forward_auc_silencioso(X_var, y)
        auc_var_prom = np.mean(aucs_var)
        std_v = np.std(aucs_var)
        dif = auc_var_prom - auc_base_prom
        resultados[feat] = {'auc': auc_var_prom, 'std': std_v, 'dif': dif}
        flag = "✅" if dif > 0.003 else ("➖" if dif > 0 else "⚠️")
        print(f"  +{feat:<27} {auc_var_prom:>8.4f} {dif:>+9.4f} "
              f"{std_v:>12.4f} {flag}")

    # ── Las tres juntas ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Las tres features nuevas juntas")
    print(f"{'='*60}")
    X_todas = df_ml[FEATURES_BASE_ACTUALIZADO + features_nuevas]
    aucs_todas = walk_forward_auc_silencioso(X_todas, y)
    auc_todas_prom = np.mean(aucs_todas)
    print(f"  AUC promedio: {auc_todas_prom:.4f} "
          f"(vs base: {auc_todas_prom - auc_base_prom:+.4f})")

    # ── Resumen final ────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    print(f"  Baseline:                        {auc_base_prom:.4f}")
    for feat, r in resultados.items():
        print(f"  + {feat:<30}: {r['auc']:.4f} ({r['dif']:+.4f})")
    print(f"  + las 3 juntas:                     {auc_todas_prom:.4f} "
          f"({auc_todas_prom-auc_base_prom:+.4f})")

    mejor = max(resultados, key=lambda f: resultados[f]['auc'])
    print(f"\n  Mejor feature individual: {mejor} "
          f"(AUC={resultados[mejor]['auc']:.4f})")