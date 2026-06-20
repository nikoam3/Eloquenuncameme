import sys
import os
# Encuentra la carpeta principal 'backtesting' subiendo un nivel
ruta_principal = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ruta_principal)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_DATASET = 'dataset_ml.parquet'
UMBRAL_PROB     = 0.75
N_VENTANAS      = 4

FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx', 'rsi_pendiente_vs_precio',
    'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


# ============================================================
# PASO 1: DIAGNÓSTICO — ¿Qué tan mal está calibrado?
# ============================================================
def diagnosticar_calibracion(X: pd.DataFrame, y: pd.Series):
    """
    Mide la calibración del modelo actual SIN calibrar.
    Usamos cross_val_predict para obtener probabilidades
    out-of-sample (sin data leakage).
    """
    print("=" * 55)
    print("  DIAGNÓSTICO DE CALIBRACIÓN")
    print("=" * 55)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    # Probabilidades out-of-sample (5 folds cronológicos)
    # Usamos cv=5 con shuffle=False para respetar el tiempo
    probs_oos = cross_val_predict(
        modelo, X_scaled, y,
        cv=5,
        method='predict_proba'
    )[:, 1]

    # ── Reliability curve ───────────────────────────────────
    prob_true, prob_pred = calibration_curve(
        y, probs_oos, n_bins=10, strategy='uniform'
    )

    # ── Métricas de calibración ─────────────────────────────
    # Expected Calibration Error (ECE): cuánto se desvía en promedio
    ece = np.mean(np.abs(prob_true - prob_pred))

    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  (0.00 = perfecto | >0.05 = problema real)")

    print(f"\n  Detalle por bin de probabilidad:")
    print(f"  {'Pred (modelo)':>15} {'Real (datos)':>14} {'Diferencia':>12}")
    print(f"  {'─'*45}")
    for pp, pt in zip(prob_pred, prob_true):
        diff = pt - pp
        estado = "⚠️" if abs(diff) > 0.05 else "✅"
        print(f"  {pp:>14.3f} {pt:>13.3f} {diff:>+11.3f}  {estado}")

    # ── Distribución de probabilidades ──────────────────────
    print(f"\n  Distribución de probabilidades predichas:")
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    hist, edges = np.histogram(probs_oos, bins=bins)
    total = len(probs_oos)
    for i, count in enumerate(hist):
        bar = "█" * int(count / total * 40)
        print(f"  [{edges[i]:.1f}-{edges[i+1]:.1f}]: {bar} {count:,} ({count/total*100:.1f}%)")

    # ── Análisis del umbral actual ───────────────────────────
    mask_umbral = probs_oos >= UMBRAL_PROB
    n_trades    = mask_umbral.sum()
    if n_trades > 0:
        precision_real = y[mask_umbral].mean()
        print(f"\n  Con umbral {UMBRAL_PROB}:")
        print(f"    Trades seleccionados: {n_trades:,} ({n_trades/total*100:.1f}%)")
        print(f"    Precisión real:       {precision_real:.3f} ({precision_real*100:.1f}%)")
        print(f"    Precisión esperada:   {UMBRAL_PROB:.3f} ({UMBRAL_PROB*100:.1f}%)")
        gap = precision_real - UMBRAL_PROB
        print(f"    Gap (real - esperada): {gap:+.3f}")
        if gap < -0.05:
            print(f"    ⚠️  El modelo es SOBRECONFIADO en este rango")
        elif gap > 0.05:
            print(f"    ⚠️  El modelo es SUBCONFIADO en este rango")
        else:
            print(f"    ✅  Calibración aceptable en este rango")

    return probs_oos, prob_true, prob_pred, ece


# ============================================================
# PASO 2: CALIBRACIÓN — Platt Scaling vs Isotonic
# ============================================================
def comparar_calibradores(X: pd.DataFrame, y: pd.Series):
    """
    Compara el modelo sin calibrar vs dos métodos de calibración.
    Usamos walk-forward para no contaminar los datos.
    """
    print("\n" + "=" * 55)
    print("  COMPARACIÓN DE CALIBRADORES (walk-forward)")
    print("=" * 55)

    n          = len(X)
    tam_test   = n // (N_VENTANAS + 1)
    resultados = {'sin_calibrar': [], 'platt': [], 'isotonic': []}

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

        modelo_base = LogisticRegression(
            max_iter=1000, class_weight='balanced'
        )

        # Sin calibrar
        modelo_base.fit(X_tr_sc, y_train)
        p_base = modelo_base.predict_proba(X_te_sc)[:, 1]

        # Platt Scaling (sigmoid)
        modelo_platt = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            method='sigmoid', cv=3
        )
        modelo_platt.fit(X_tr_sc, y_train)
        p_platt = modelo_platt.predict_proba(X_te_sc)[:, 1]

        # Isotonic Regression
        modelo_iso = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            method='isotonic', cv=3
        )
        modelo_iso.fit(X_tr_sc, y_train)
        p_iso = modelo_iso.predict_proba(X_te_sc)[:, 1]

        # ECE de cada uno
        for nombre, probs in [('sin_calibrar', p_base),
                               ('platt',        p_platt),
                               ('isotonic',     p_iso)]:
            pt, pp = calibration_curve(y_test, probs,
                                       n_bins=10, strategy='uniform')
            ece = np.mean(np.abs(pt - pp))
            resultados[nombre].append(ece)

        fecha = X.index[fin_train].strftime('%m/%Y')
        print(f"\n  Ventana {i+1} ({fecha}):")
        for nombre, eces in resultados.items():
            if len(eces) == i + 1:
                print(f"    {nombre:<15}: ECE = {eces[-1]:.4f}")

    print(f"\n  {'─'*40}")
    print(f"  Promedio ECE:")
    for nombre, eces in resultados.items():
        avg = np.mean(eces)
        mejor = " ← MEJOR" if avg == min(np.mean(v) for v in resultados.values()) else ""
        print(f"    {nombre:<15}: {avg:.4f}{mejor}")

    # Determinar el mejor
    mejor_nombre = min(resultados, key=lambda k: np.mean(resultados[k]))
    return mejor_nombre


# ============================================================
# PASO 3: GRÁFICO DE CALIBRACIÓN
# ============================================================
def graficar_calibracion(X: pd.DataFrame, y: pd.Series,
                         mejor_calibrador: str):
    """
    Grafica el reliability diagram comparando los 3 métodos.
    """
    n        = len(X)
    fin_test = int(n * 0.8)

    X_train = X.iloc[:fin_test]
    y_train = y.iloc[:fin_test]
    X_test  = X.iloc[fin_test:]
    y_test  = y.iloc[fin_test:]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colores = {
        'Sin calibrar': 'red',
        'Platt (sigmoid)': 'blue',
        'Isotonic': 'green'
    }

    for ax_idx, (nombre, method) in enumerate([
        ('Sin calibrar', None),
        ('Platt (sigmoid)', 'sigmoid'),
        ('Isotonic', 'isotonic')
    ]):
        if method is None:
            m = LogisticRegression(max_iter=1000, class_weight='balanced')
            m.fit(X_tr_sc, y_train)
            probs = m.predict_proba(X_te_sc)[:, 1]
        else:
            m = CalibratedClassifierCV(
                LogisticRegression(max_iter=1000, class_weight='balanced'),
                method=method, cv=3
            )
            m.fit(X_tr_sc, y_train)
            probs = m.predict_proba(X_te_sc)[:, 1]

        pt, pp = calibration_curve(y_test, probs,
                                   n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(pt - pp))

        axes[0].plot(pp, pt,
                     marker='o',
                     label=f"{nombre} (ECE={ece:.3f})",
                     color=colores[nombre])

    # Línea perfecta
    axes[0].plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
    axes[0].set_xlabel('Probabilidad predicha')
    axes[0].set_ylabel('Probabilidad real (fracción positivos)')
    axes[0].set_title('Reliability Diagram')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Histograma de probabilidades
    for nombre, method in [('Sin calibrar', None),
                            ('Platt', 'sigmoid'),
                            ('Isotonic', 'isotonic')]:
        if method is None:
            m = LogisticRegression(max_iter=1000, class_weight='balanced')
            m.fit(X_tr_sc, y_train)
            probs = m.predict_proba(X_te_sc)[:, 1]
        else:
            m = CalibratedClassifierCV(
                LogisticRegression(max_iter=1000, class_weight='balanced'),
                method=method, cv=3
            )
            m.fit(X_tr_sc, y_train)
            probs = m.predict_proba(X_te_sc)[:, 1]

        axes[1].hist(probs, bins=30, alpha=0.5,
                     label=nombre, density=True)

    axes[1].axvline(UMBRAL_PROB, color='black',
                    linestyle='--', label=f'Umbral ({UMBRAL_PROB})')
    axes[1].set_xlabel('Probabilidad predicha')
    axes[1].set_ylabel('Densidad')
    axes[1].set_title('Distribución de probabilidades')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'Análisis de Calibración — Mejor: {mejor_calibrador}',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('calibracion_analisis.png', dpi=150)
    plt.show()
    print("✅ Gráfico guardado como calibracion_analisis.png")


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔧 Análisis de Calibración\n")

    df = pd.read_parquet(ARCHIVO_DATASET)
    X  = df[FEATURES]
    y  = df['target']

    print(f"Dataset: {len(df):,} filas x {len(FEATURES)} features")

    # 1. Diagnóstico
    probs_oos, prob_true, prob_pred, ece = diagnosticar_calibracion(X, y)

    # 2. Comparar calibradores
    mejor = comparar_calibradores(X, y)
    print(f"\n  → Mejor calibrador: {mejor}")

    # 3. Gráfico
    graficar_calibracion(X, y, mejor)