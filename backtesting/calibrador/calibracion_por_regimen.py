import sys
import os
# Encuentra la carpeta principal 'backtesting' subiendo un nivel
ruta_principal = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ruta_principal)
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_DATASET = 'dataset_ml.parquet'
N_VENTANAS      = 4
UMBRAL_ADX      = 25  # criterio de régimen

FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx', 'rsi_pendiente_vs_precio',
    'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


def ece_de(y_true, y_prob, n_bins=10):
    """Calcula el ECE de un conjunto de predicciones."""
    if len(y_true) < n_bins * 2:
        return np.nan  # muy pocos datos para medir bien
    pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return np.mean(np.abs(pt - pp))


def evaluar_ventana_con_regimen(X: pd.DataFrame, y: pd.Series,
                                fin_train: int, fin_test: int, nombre: str):
    """
    Para una ventana de walk-forward:
      1. Entrena UN modelo base (igual que siempre)
      2. Mide ECE global (sin separar régimen) - baseline
      3. Calibra por separado en tendencia vs lateral
      4. Mide ECE combinado con calibración por régimen
    """
    X_train = X.iloc[:fin_train]
    y_train = y.iloc[:fin_train]
    X_test  = X.iloc[fin_train:fin_test]
    y_test  = y.iloc[fin_train:fin_test]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # ── Modelo base (sin calibrar) ──────────────────────────
    modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo.fit(X_tr_sc, y_train)
    probs_test_base = modelo.predict_proba(X_te_sc)[:, 1]

    ece_global_sin_calibrar = ece_de(y_test, probs_test_base)

    # ── Máscaras de régimen en TRAIN y TEST ─────────────────
    mask_tendencia_train = (X_train['adx'] > UMBRAL_ADX).values
    mask_lateral_train   = ~mask_tendencia_train

    mask_tendencia_test = (X_test['adx'] > UMBRAL_ADX).values
    mask_lateral_test   = ~mask_tendencia_test

    n_tendencia_test = mask_tendencia_test.sum()
    n_lateral_test   = mask_lateral_test.sum()

    # ── Calibrador único (Isotonic, todo el train) ──────────
    cal_unico = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        method='isotonic', cv=3
    )
    cal_unico.fit(X_tr_sc, y_train)
    probs_unico = cal_unico.predict_proba(X_te_sc)[:, 1]
    ece_unico = ece_de(y_test, probs_unico)

    # ── Calibrador POR RÉGIMEN ──────────────────────────────
    # Entrenamos un calibrador separado para cada régimen,
    # usando solo los datos de train de ese régimen.
    probs_por_regimen = np.zeros(len(y_test))

    for mask_train, mask_test, etiqueta in [
        (mask_tendencia_train, mask_tendencia_test, 'tendencia'),
        (mask_lateral_train,   mask_lateral_test,   'lateral')
    ]:
        if mask_train.sum() < 50 or mask_test.sum() < 10:
            # Muy pocos datos en este régimen, usamos el calibrador único
            probs_por_regimen[mask_test] = probs_unico[mask_test]
            continue

        cal_regimen = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            method='isotonic', cv=3
        )
        cal_regimen.fit(X_tr_sc[mask_train], y_train[mask_train])
        probs_por_regimen[mask_test] = cal_regimen.predict_proba(
            X_te_sc[mask_test]
        )[:, 1]

    ece_por_regimen = ece_de(y_test, probs_por_regimen)

    # ── ECE específico de cada régimen (para detalle) ───────
    ece_tendencia_unico = ece_de(y_test[mask_tendencia_test],
                                  probs_unico[mask_tendencia_test])
    ece_lateral_unico   = ece_de(y_test[mask_lateral_test],
                                  probs_unico[mask_lateral_test])

    ece_tendencia_regimen = ece_de(y_test[mask_tendencia_test],
                                    probs_por_regimen[mask_tendencia_test])
    ece_lateral_regimen   = ece_de(y_test[mask_lateral_test],
                                    probs_por_regimen[mask_lateral_test])

    print(f"\n{'='*65}")
    print(f"  {nombre}")
    print(f"{'='*65}")
    print(f"  Tamaño test: {len(y_test):,}  "
          f"(tendencia: {n_tendencia_test:,} | lateral: {n_lateral_test:,})")
    print(f"\n  {'Método':<28} {'ECE global':>12}")
    print(f"  {'─'*42}")
    print(f"  {'Sin calibrar':<28} {ece_global_sin_calibrar:>12.4f}")
    print(f"  {'Isotonic único':<28} {ece_unico:>12.4f}")
    print(f"  {'Isotonic por régimen':<28} {ece_por_regimen:>12.4f}")

    print(f"\n  Detalle por régimen (Isotonic único vs por régimen):")
    print(f"  {'Régimen':<14} {'Único':>10} {'Por régimen':>14}")
    print(f"  {'─'*40}")
    print(f"  {'Tendencia':<14} {ece_tendencia_unico:>10.4f} "
          f"{ece_tendencia_regimen:>14.4f}")
    print(f"  {'Lateral':<14} {ece_lateral_unico:>10.4f} "
          f"{ece_lateral_regimen:>14.4f}")

    return {
        'nombre': nombre,
        'sin_calibrar': ece_global_sin_calibrar,
        'unico': ece_unico,
        'por_regimen': ece_por_regimen
    }


if __name__ == "__main__":
    print("🔧 Calibración por régimen de mercado (ADX)\n")

    df = pd.read_parquet(ARCHIVO_DATASET)
    X  = df[FEATURES]
    y  = df['target']

    print(f"Dataset: {len(df):,} filas")
    print(f"Criterio de régimen: ADX > {UMBRAL_ADX} = tendencia, "
          f"ADX <= {UMBRAL_ADX} = lateral")

    n        = len(X)
    tam_test = n // (N_VENTANAS + 1)

    resultados = []
    for i in range(N_VENTANAS):
        fin_train = tam_test * (i + 1)
        fin_test  = tam_test * (i + 2)
        r = evaluar_ventana_con_regimen(X, y, fin_train, fin_test, f"V{i+1}")
        resultados.append(r)

    print(f"\n\n{'='*65}")
    print("  RESUMEN FINAL")
    print(f"{'='*65}")
    print(f"  {'Ventana':<10} {'Sin calibrar':>14} {'Único':>10} {'Por régimen':>14}")
    for r in resultados:
        print(f"  {r['nombre']:<10} {r['sin_calibrar']:>14.4f} "
              f"{r['unico']:>10.4f} {r['por_regimen']:>14.4f}")

    print(f"\n  Promedios:")
    print(f"    Sin calibrar:    {np.mean([r['sin_calibrar'] for r in resultados]):.4f}")
    print(f"    Isotonic único:  {np.mean([r['unico'] for r in resultados]):.4f}")
    print(f"    Por régimen:     {np.mean([r['por_regimen'] for r in resultados]):.4f}")

"""
Si la hipótesis del régimen es correcta, 
deberíamos ver que "Por régimen" mejora el ECE 
especialmente en V1 y V2, las ventanas problemáticas, 
más que en V3/V4 que ya estaban relativamente bien.
"""