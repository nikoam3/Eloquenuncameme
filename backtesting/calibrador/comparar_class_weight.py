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
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_DATASET = 'dataset_ml.parquet'
N_VENTANAS      = 4
UMBRAL_PROB     = 0.63

FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200', 'ema9_vs_ema26',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx', 'rsi_pendiente_vs_precio',
    'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


def ece_de(y_true, y_prob, n_bins=10):
    if len(y_true) < n_bins * 2:
        return np.nan
    pt, pp = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    return np.mean(np.abs(pt - pp))


def evaluar_configuracion(X_tr_sc, y_train, X_te_sc, y_test,
                           class_weight, calibrar, etiqueta):
    """
    Entrena un modelo con la configuración dada y mide:
    - ECE (calibración)
    - Precisión real con umbral 0.63
    - AUC (capacidad discriminativa, no depende del umbral)
    - % de señales generadas (frecuencia de trading)
    """
    if calibrar:
        modelo = CalibratedClassifierCV(
            LogisticRegression(
                max_iter=1000, class_weight=class_weight
            ),
            method='isotonic', cv=3
        )
    else:
        modelo = LogisticRegression(
            max_iter=1000, class_weight=class_weight
        )

    modelo.fit(X_tr_sc, y_train)
    probs = modelo.predict_proba(X_te_sc)[:, 1]

    ece = ece_de(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    mask = probs >= UMBRAL_PROB
    n_señales = mask.sum()
    precision_real = y_test[mask].mean() if n_señales > 0 else 0
    pct_señales = n_señales / len(y_test) * 100

    return {
        'etiqueta': etiqueta,
        'ece': ece,
        'auc': auc,
        'precision_real': precision_real,
        'pct_señales': pct_señales,
        'n_señales': n_señales
    }


def walk_forward_comparacion(X: pd.DataFrame, y: pd.Series):
    """
    Compara 4 configuraciones en walk-forward:
      A) balanced + sin calibrar   (tu situación actual)
      B) balanced + con calibrar   (isotonic)
      C) sin balanced + sin calibrar
      D) sin balanced + con calibrar
    """
    n        = len(X)
    tam_test = n // (N_VENTANAS + 1)

    configs = [
        ('A: balanced  + sin calibrar', 'balanced', False),
        ('B: balanced  + con calibrar', 'balanced', True),
        ('C: sin bal.  + sin calibrar', None,       False),
        ('D: sin bal.  + con calibrar', None,       True),
    ]

    # Acumulamos métricas por configuración
    acum = {cfg[0]: {'ece': [], 'auc': [], 'prec': [], 'pct': []}
            for cfg in configs}

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

        fecha = X.index[fin_train].strftime('%m/%Y')
        print(f"\n{'='*70}")
        print(f"  V{i+1} ({fecha}) — test: {len(y_test):,} filas "
              f"| balance: {y_test.mean()*100:.1f}% positivos")
        print(f"{'='*70}")
        print(f"  {'Config':<30} {'ECE':>7} {'AUC':>7} "
              f"{'Prec@0.63':>10} {'% señales':>10}")
        print(f"  {'─'*65}")

        for etiqueta, cw, calibrar in configs:
            r = evaluar_configuracion(
                X_tr_sc, y_train, X_te_sc, y_test,
                cw, calibrar, etiqueta
            )
            acum[etiqueta]['ece'].append(r['ece'])
            acum[etiqueta]['auc'].append(r['auc'])
            acum[etiqueta]['prec'].append(r['precision_real'])
            acum[etiqueta]['pct'].append(r['pct_señales'])

            print(f"  {etiqueta:<30} {r['ece']:>7.4f} {r['auc']:>7.4f} "
                  f"{r['precision_real']:>9.3f} {r['pct_señales']:>9.1f}%")

    # Resumen final
    print(f"\n\n{'='*70}")
    print("  PROMEDIO WALK-FORWARD (4 ventanas)")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'ECE':>7} {'AUC':>7} "
          f"{'Prec@0.63':>10} {'% señales':>10}")
    print(f"  {'─'*65}")

    for etiqueta, _, _ in configs:
        a = acum[etiqueta]
        print(f"  {etiqueta:<30} "
              f"{np.mean(a['ece']):>7.4f} "
              f"{np.mean(a['auc']):>7.4f} "
              f"{np.mean(a['prec']):>9.3f} "
              f"{np.mean(a['pct']):>9.1f}%")

    print(f"\n  Interpretación:")
    print(f"  • ECE   → más bajo es mejor (0 = calibración perfecta)")
    print(f"  • AUC   → más alto es mejor (capacidad de ordenar bien)")
    print(f"  • Prec  → precisión real con umbral {UMBRAL_PROB}")
    print(f"  • %señ  → qué % de velas generan señal de compra")


if __name__ == "__main__":
    print("⚖️  Comparación: class_weight='balanced' vs sin balanceo\n")

    df = pd.read_parquet(ARCHIVO_DATASET)
    X  = df[FEATURES]
    y  = df['target']

    print(f"Dataset: {len(df):,} filas")
    print(f"Balance global: {y.mean()*100:.1f}% positivos")

    walk_forward_comparacion(X, y)