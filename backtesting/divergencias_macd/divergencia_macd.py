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
PAR             = "ETHUSDT"
STOP_LOSS       = 0.025
TAKE_PROFIT     = 0.010
MAX_VELAS       = 96
N_VENTANAS      = 4
VENTANA_PENDIENTE = 20  # la misma que ganó con RSI, punto de partida

FEATURES_BASE = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h',
    'rsi_pendiente_vs_precio'
]

def calcular_macd_features(df: pd.DataFrame,
                           ventana_pendiente: int = VENTANA_PENDIENTE) -> pd.DataFrame:
    """
    Calcula tres variantes de MACD para evaluar cuál (si alguna)
    aporta señal real, aplicando la lección de 7B.1: priorizamos
    versiones continuas sobre binarias.
    """
    df = df.copy()

    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']      = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist']   = macd['MACDh_12_26_9']

    # Variante 1: histograma normalizado por precio (para que sea
    # comparable entre distintos niveles de precio de ETH en el tiempo)
    df['macd_histograma'] = df['MACD_hist'] / df['Close'] * 1000

    # Variante 2: "pendiente vs precio", análogo a rsi_pendiente_vs_precio
    # pero usando el MACD en vez del RSI
    cambio_precio_pct = (
        df['Close'] / df['Close'].shift(ventana_pendiente) - 1
    ) * 100
    cambio_macd = df['MACD'] - df['MACD'].shift(ventana_pendiente)
    # Normalizamos el cambio de MACD por precio, mismo motivo que arriba
    cambio_macd_norm = cambio_macd / df['Close'] * 1000
    df['macd_pendiente_vs_precio'] = cambio_precio_pct - cambio_macd_norm

    # Variante 3: cruce de MACD sobre su señal (evento discreto,
    # la incluimos a pesar de ser binaria porque mide algo DISTINTO
    # a una divergencia: un cambio de régimen de momentum, no una
    # comparación de extremos)
    cruce_alcista = (
        (df['MACD'] > df['MACD_signal']) &
        (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    )
    cruce_bajista = (
        (df['MACD'] < df['MACD_signal']) &
        (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    )
    df['macd_cruce'] = 0
    df.loc[cruce_alcista, 'macd_cruce'] = 1
    df.loc[cruce_bajista, 'macd_cruce'] = -1

    return df


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
    print("🔬 Evaluación de features MACD (Módulo 7B.2)\n")

    print("Cargando datos y calculando RSI + MACD...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    df_15m = df_15m.dropna()
    df_15m = calcular_macd_features(df_15m)

    print("Construyendo features base (multi-timeframe)...")
    features_df, df_15m_completo = construir_features("ETHUSDT")
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, max_velas=MAX_VELAS
    )

    cols_macd = ['macd_histograma', 'macd_pendiente_vs_precio',
                 'macd_cruce']
    features_macd_alineadas = df_15m[cols_macd].reindex(features_df.index)

    df_ml = features_df.join(target).join(features_macd_alineadas).dropna()
    print(f"Dataset final: {len(df_ml):,} filas\n")

    y = df_ml['target']

    # ── Nuevo baseline: base + rsi_pendiente (ya confirmada) ──
    print(f"{'='*60}")
    print("  Baseline NUEVO (incluye rsi_pendiente_vs_precio)")
    print(f"{'='*60}")
    X_base = df_ml[FEATURES_BASE]
    aucs_base = walk_forward_auc_silencioso(X_base, y)
    auc_base_prom = np.mean(aucs_base)
    print(f"  AUC promedio: {auc_base_prom:.4f}")
    print(f"  Por ventana: {[round(a, 4) for a in aucs_base]}")

    # ── Cada variante de MACD, individual ───────────────────
    print(f"\n{'='*60}")
    print("  Aporte individual de cada variante de MACD")
    print(f"{'='*60}")
    print(f"  {'Feature agregada':<28} {'AUC':>8} {'vs base':>10}")
    print(f"  {'─'*48}")

    variantes_macd = ['macd_histograma', 'macd_pendiente_vs_precio', 'macd_cruce']
    resultados = {}

    for feat in variantes_macd:
        cols = FEATURES_BASE + [feat]
        X_var = df_ml[cols]
        aucs_var = walk_forward_auc_silencioso(X_var, y)
        auc_var_prom = np.mean(aucs_var)
        dif = auc_var_prom - auc_base_prom
        resultados[feat] = auc_var_prom
        flag = "✅" if dif > 0.003 else ("➖" if dif > 0 else "⚠️")
        print(f"  +{feat:<27} {auc_var_prom:>8.4f} {dif:>+9.4f} {flag}")

    # ── Las tres variantes juntas ─────────────────────────────
    print(f"\n{'='*60}")
    print("  Las tres variantes de MACD juntas")
    print(f"{'='*60}")
    X_todas = df_ml[FEATURES_BASE + variantes_macd]
    aucs_todas = walk_forward_auc_silencioso(X_todas, y)
    auc_todas_prom = np.mean(aucs_todas)
    print(f"  AUC promedio: {auc_todas_prom:.4f} "
          f"(vs base: {auc_todas_prom - auc_base_prom:+.4f})")

    print(f"\n\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    print(f"  Baseline (con rsi_pendiente):  {auc_base_prom:.4f}")
    for feat, auc in resultados.items():
        print(f"  + {feat:<28}: {auc:.4f} ({auc-auc_base_prom:+.4f})")
    print(f"  + las 3 juntas:                  {auc_todas_prom:.4f} "
          f"({auc_todas_prom-auc_base_prom:+.4f})")