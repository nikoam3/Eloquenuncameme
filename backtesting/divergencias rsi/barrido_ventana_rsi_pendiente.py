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
PAR          = "ETHUSDT"
STOP_LOSS    = 0.025
TAKE_PROFIT  = 0.010
MAX_VELAS    = 96
N_VENTANAS   = 4

VENTANAS_A_PROBAR = [5, 8, 10, 14, 20, 25, 30, 40, 50]

FEATURES_BASE = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


def calcular_rsi_pendiente_vs_precio(df: pd.DataFrame, ventana: int) -> pd.Series:
    """
    Calcula SOLO la feature ganadora, para una ventana dada.
    Misma fórmula que en divergencias.py, aislada para poder
    barrer muchos valores de ventana rápidamente.
    """
    cambio_precio_pct = (
        df['Close'] / df['Close'].shift(ventana) - 1
    ) * 100
    cambio_rsi = df['RSI'] - df['RSI'].shift(ventana)
    return cambio_precio_pct - cambio_rsi


def walk_forward_auc_silencioso(X: pd.DataFrame, y: pd.Series) -> list:
    """Misma lógica de siempre, sin prints por ventana."""
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
    print("🔬 Barrido de ventana para rsi_pendiente_vs_precio\n")

    print("Cargando datos base...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    df_15m = df_15m.dropna()

    print("Construyendo features base (multi-timeframe)...")
    features_df, df_15m_completo = construir_features("ETHUSDT")
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, max_velas=MAX_VELAS
    )

    y = target.reindex(features_df.index)

    # ── Baseline (sin la feature nueva) ─────────────────────
    print(f"\n{'='*60}")
    print("  Baseline (solo features base, sin rsi_pendiente)")
    print(f"{'='*60}")
    df_base = features_df.join(target).dropna()
    X_base  = df_base[FEATURES_BASE]
    y_base  = df_base['target']
    aucs_base = walk_forward_auc_silencioso(X_base, y_base)
    auc_base_prom = np.mean(aucs_base)
    print(f"  AUC promedio: {auc_base_prom:.4f}")
    print(f"  Por ventana: {[round(a, 4) for a in aucs_base]}")

    # ── Barrido de ventanas ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  Barrido de VENTANA para rsi_pendiente_vs_precio")
    print(f"{'='*60}")
    print(f"  {'Ventana':>8} {'AUC prom':>10} {'vs base':>10}  "
          f"{'AUC por ventana (V1,V2,V3,V4)'}")
    print(f"  {'─'*75}")

    resultados = []
    for ventana in VENTANAS_A_PROBAR:
        df_15m[f'rsi_pendiente_v{ventana}'] = calcular_rsi_pendiente_vs_precio(
            df_15m, ventana
        )

        feat_alineada = df_15m[f'rsi_pendiente_v{ventana}'].reindex(
            features_df.index
        )
        df_ml = features_df.join(target).join(
            feat_alineada.rename('rsi_pendiente')
        ).dropna()

        cols = FEATURES_BASE + ['rsi_pendiente']
        X_var = df_ml[cols]
        y_var = df_ml['target']

        aucs_var = walk_forward_auc_silencioso(X_var, y_var)
        auc_var_prom = np.mean(aucs_var)
        dif = auc_var_prom - auc_base_prom

        resultados.append({
            'ventana': ventana,
            'auc_promedio': auc_var_prom,
            'diferencia': dif,
            'aucs': aucs_var,
            'n_filas': len(df_ml)
        })

        aucs_str = ",".join(f"{a:.3f}" for a in aucs_var)
        flag = "✅" if dif > 0.01 else ("➖" if dif > 0 else "⚠️")
        print(f"  {ventana:>8} {auc_var_prom:>10.4f} {dif:>+9.4f}  "
              f"[{aucs_str}] {flag}")

    # ── Resumen y mejor ventana ──────────────────────────────
    print(f"\n\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")

    mejor = max(resultados, key=lambda r: r['auc_promedio'])
    print(f"  Mejor ventana: {mejor['ventana']}")
    print(f"  AUC promedio:  {mejor['auc_promedio']:.4f} "
          f"(vs base: {mejor['diferencia']:+.4f})")
    print(f"  AUC por ventana de walk-forward: "
          f"{[round(a, 4) for a in mejor['aucs']]}")

    # Consistencia: ¿el ganador es estable o un pico aislado?
    print(f"\n  Consistencia entre ventanas vecinas:")
    for r in resultados:
        marca = " ← MEJOR" if r['ventana'] == mejor['ventana'] else ""
        # Desviación estándar entre las 4 ventanas de walk-forward
        std_entre_ventanas = np.std(r['aucs'])
        print(f"    ventana={r['ventana']:>3}: "
              f"AUC={r['auc_promedio']:.4f}  "
              f"std={std_entre_ventanas:.4f}{marca}")