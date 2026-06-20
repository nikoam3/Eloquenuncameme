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
VENTANA_PENDIENTE = 20

FEATURES_BASE = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'rsi_pendiente_vs_precio',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]

# Combinaciones (fast, slow, signal) a probar.
# 12/26/9 es el estándar de manual, incluido como referencia.
# El resto explora alrededor: más rápido, más lento, y una
# combinación "media" entre ambos extremos.
COMBINACIONES_MACD = [
    (8, 17, 9),    # más rápido (reacciona antes)
    (12, 26, 9),   # estándar de manual
    (12, 26, 6),   # estándar pero señal más rápida
    (16, 34, 9),   # más lento (menos ruido, más rezago)
    (5, 35, 5),    # muy rápido vs muy lento (más sensible)
]


def calcular_rsi_pendiente(df: pd.DataFrame,
                           ventana: int = VENTANA_PENDIENTE) -> pd.Series:
    """Recalcula la feature ya confirmada, para que esté en el baseline."""
    cambio_precio_pct = (
        df['Close'] / df['Close'].shift(ventana) - 1
    ) * 100
    cambio_rsi = df['RSI'] - df['RSI'].shift(ventana)
    return cambio_precio_pct - cambio_rsi


def calcular_macd_histograma(df: pd.DataFrame,
                             fast: int, slow: int, signal: int,
                             normalizacion: str = 'precio') -> pd.Series:
    """
    Calcula el histograma de MACD con los parámetros dados,
    normalizado según el método elegido:
      - 'precio': histograma / Close * 1000  (la que ya probamos)
      - 'atr':    histograma / ATR            (relativo a volatilidad)
      - 'crudo':  sin normalizar (valor absoluto del histograma)
    """
    macd = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
    col_hist = f'MACDh_{fast}_{slow}_{signal}'
    hist = macd[col_hist]

    if normalizacion == 'precio':
        return hist / df['Close'] * 1000
    elif normalizacion == 'atr':
        # ATR ya viene calculado en df si se llamó antes a
        # ta.atr(); si no, lo calculamos acá
        if 'ATR' not in df.columns:
            df_temp_atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            return hist / df_temp_atr
        return hist / df['ATR']
    elif normalizacion == 'crudo':
        return hist
    else:
        raise ValueError(f"Normalización desconocida: {normalizacion}")


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


def preparar_dataset_completo(df_15m: pd.DataFrame):
    """Prepara features base + target, una sola vez, reutilizable."""
    features_df, df_15m_completo = construir_features("ETHUSDT")
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, max_velas=MAX_VELAS
    )

    #df_15m['rsi_pendiente_vs_precio'] = calcular_rsi_pendiente(df_15m)

    return features_df, target


def evaluar_macd_variante(features_df, target, df_15m,
                          fast, slow, signal, normalizacion):
    """Calcula UNA variante de MACD y mide su AUC incremental."""
    hist = calcular_macd_histograma(df_15m, fast, slow, signal, normalizacion)
    nombre_col = f'macd_hist_{fast}_{slow}_{signal}_{normalizacion}'
    df_15m_local = df_15m.copy()
    df_15m_local[nombre_col] = hist

    feat_alineada = df_15m_local[nombre_col].reindex(features_df.index)
    df_ml = features_df.join(target).join(feat_alineada).dropna()

    cols = FEATURES_BASE + [nombre_col]
    X = df_ml[cols]
    y = df_ml['target']

    aucs = walk_forward_auc_silencioso(X, y)
    return aucs, len(df_ml)


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Barrido de parámetros y normalización de MACD\n")

    print("Cargando datos base...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    df_15m['ATR'] = ta.atr(df_15m['High'], df_15m['Low'],
                            df_15m['Close'], length=14)
    df_15m = df_15m.dropna()

    print("Construyendo features base + target...")
    features_df, target = preparar_dataset_completo(df_15m)

    # ── Baseline (sin MACD) ──────────────────────────────────
    print(f"\n{'='*65}")
    print("  Baseline (features base + rsi_pendiente, sin MACD)")
    print(f"{'='*65}")
    df_base = features_df.join(target).dropna()
    X_base  = df_base[FEATURES_BASE]
    y_base  = df_base['target']
    aucs_base = walk_forward_auc_silencioso(X_base, y_base)
    auc_base_prom = np.mean(aucs_base)
    print(f"  AUC promedio: {auc_base_prom:.4f}")

    # ── BARRIDO 1: parámetros (fast, slow, signal) ──────────
    print(f"\n{'='*65}")
    print("  BARRIDO 1: Parámetros del MACD (normalización fija: /precio)")
    print(f"{'='*65}")
    print(f"  {'(fast,slow,signal)':<20} {'AUC':>8} {'vs base':>10} "
          f"{'std entre V':>12}")
    print(f"  {'─'*55}")

    resultados_parametros = []
    for fast, slow, signal in COMBINACIONES_MACD:
        aucs, n = evaluar_macd_variante(
            features_df, target, df_15m,
            fast, slow, signal, normalizacion='precio'
        )
        auc_prom = np.mean(aucs)
        std_v    = np.std(aucs)
        dif      = auc_prom - auc_base_prom
        resultados_parametros.append({
            'combo': (fast, slow, signal),
            'auc': auc_prom, 'std': std_v, 'dif': dif
        })
        flag = "✅" if dif > 0.005 else ("➖" if dif > 0 else "⚠️")
        print(f"  ({fast},{slow},{signal})"
              f"{'':<{20-len(f'({fast},{slow},{signal})')}} "
              f"{auc_prom:>8.4f} {dif:>+9.4f} {std_v:>12.4f} {flag}")

    mejor_combo = max(resultados_parametros, key=lambda r: r['auc'])
    fast_m, slow_m, signal_m = mejor_combo['combo']
    print(f"\n  → Mejor combinación: ({fast_m},{slow_m},{signal_m}) "
          f"con AUC={mejor_combo['auc']:.4f}")

    # ── BARRIDO 2: normalización (usando los mejores parámetros) ──
    print(f"\n{'='*65}")
    print(f"  BARRIDO 2: Normalización (parámetros fijos: "
          f"{fast_m}/{slow_m}/{signal_m})")
    print(f"{'='*65}")
    print(f"  {'Normalización':<16} {'AUC':>8} {'vs base':>10} "
          f"{'std entre V':>12}")
    print(f"  {'─'*50}")

    resultados_normalizacion = []
    for norm in ['precio', 'atr', 'crudo']:
        aucs, n = evaluar_macd_variante(
            features_df, target, df_15m,
            fast_m, slow_m, signal_m, normalizacion=norm
        )
        auc_prom = np.mean(aucs)
        std_v    = np.std(aucs)
        dif      = auc_prom - auc_base_prom
        resultados_normalizacion.append({
            'norm': norm, 'auc': auc_prom, 'std': std_v, 'dif': dif
        })
        flag = "✅" if dif > 0.005 else ("➖" if dif > 0 else "⚠️")
        print(f"  {norm:<16} {auc_prom:>8.4f} {dif:>+9.4f} "
              f"{std_v:>12.4f} {flag}")

    mejor_norm = max(resultados_normalizacion, key=lambda r: r['auc'])

    # ── RESUMEN FINAL ─────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  RESUMEN FINAL")
    print(f"{'='*65}")
    print(f"  Baseline (sin MACD):           {auc_base_prom:.4f}")
    print(f"  Mejor combo parámetros:        ({fast_m},{slow_m},{signal_m}) "
          f"→ {mejor_combo['auc']:.4f}")
    print(f"  Mejor normalización:           {mejor_norm['norm']} "
          f"→ {mejor_norm['auc']:.4f}")
    print(f"\n  Configuración final recomendada:")
    print(f"    MACD({fast_m}, {slow_m}, {signal_m}), "
          f"normalizado por '{mejor_norm['norm']}'")
    print(f"    AUC final: {mejor_norm['auc']:.4f} "
          f"(mejora de {mejor_norm['auc'] - auc_base_prom:+.4f} "
          f"sobre el baseline)")