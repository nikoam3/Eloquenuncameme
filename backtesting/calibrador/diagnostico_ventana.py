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
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_DATASET = 'dataset_ml.parquet'
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


def caracterizar_ventana(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                          inicio: int, fin: int, nombre: str):
    """
    Describe las condiciones de mercado de una ventana específica:
    volatilidad, balance de clases, distribución de features clave.
    """
    d = df.iloc[inicio:fin]
    target_v = y.iloc[inicio:fin]

    print(f"\n{'─'*60}")
    print(f"  {nombre}  ({d.index[0].strftime('%Y-%m-%d')} → "
          f"{d.index[-1].strftime('%Y-%m-%d')})")
    print(f"{'─'*60}")

    # Balance de clases
    pos = target_v.mean()
    print(f"  Balance target:        {pos*100:.1f}% positivos "
          f"({len(target_v):,} filas)")

    # Volatilidad (ATR relativo promedio)
    if 'atr_relativo' in d.columns:
        atr_avg = d['atr_relativo'].mean()
        atr_std = d['atr_relativo'].std()
        print(f"  ATR relativo:          media={atr_avg:.4f} "
              f"std={atr_std:.4f}")

    # Tendencia (ADX)
    if 'adx' in d.columns:
        adx_avg = d['adx'].mean()
        pct_tendencia = (d['adx'] > 25).mean() * 100
        print(f"  ADX promedio:          {adx_avg:.1f}  "
              f"(% velas en tendencia fuerte: {pct_tendencia:.1f}%)")

    # Tendencia 1h
    if 'tendencia_1h' in d.columns:
        pct_alcista = d['tendencia_1h'].mean() * 100
        print(f"  % tiempo en alza (1h): {pct_alcista:.1f}%")

    # RSI promedio
    if 'rsi' in d.columns:
        print(f"  RSI promedio:          {d['rsi'].mean():.1f}  "
              f"(std={d['rsi'].std():.1f})")

    return {
        'nombre': nombre,
        'balance': pos,
        'atr_avg': atr_avg if 'atr_relativo' in d.columns else None,
        'adx_avg': adx_avg if 'adx' in d.columns else None,
    }


def comparar_distribuciones_train_test(X: pd.DataFrame, y: pd.Series,
                                       fin_train: int, fin_test: int,
                                       nombre: str):
    """
    Compara la distribución de features entre el set de entrenamiento
    y el de test de cada ventana. Si son muy distintas, hay
    'data drift' -> el modelo entrenado en el pasado no generaliza
    bien al futuro de esa ventana.
    """
    X_train = X.iloc[:fin_train]
    X_test  = X.iloc[fin_train:fin_test]

    print(f"\n  Drift de features (train vs test) — {nombre}:")
    print(f"  {'Feature':<20} {'Media train':>12} {'Media test':>12} "
          f"{'Dif (std)':>10}")
    print(f"  {'─'*56}")

    for col in FEATURES:
        mean_train = X_train[col].mean()
        mean_test  = X_test[col].mean()
        std_train  = X_train[col].std()
        # Diferencia normalizada en desviaciones estándar
        if std_train > 0:
            dif_std = (mean_test - mean_train) / std_train
        else:
            dif_std = 0
        flag = "⚠️" if abs(dif_std) > 0.5 else ""
        print(f"  {col:<20} {mean_train:>12.3f} {mean_test:>12.3f} "
              f"{dif_std:>+9.2f}σ {flag}")


def evaluar_calibracion_por_ventana(X: pd.DataFrame, y: pd.Series):
    """
    Para cada ventana de walk-forward, entrena y mide la calibración,
    junto con la caracterización de mercado de la ventana de TEST
    (que es la que importa, porque es donde se mide el ECE).
    """
    n        = len(X)
    tam_test = n // (N_VENTANAS + 1)

    caracteristicas = []

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

        pt, pp = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(pt - pp))

        nombre = f"V{i+1}"
        print(f"\n{'='*60}")
        print(f"  {nombre} — ECE en test: {ece:.4f}")
        print(f"{'='*60}")

        # Caracterizamos la ventana de TEST (donde se mide el ECE)
        info = caracterizar_ventana(X, X, y, fin_train, fin_test, f"{nombre} (test)")
        info['ece'] = ece
        caracteristicas.append(info)

        # Comparamos distribución train vs test (detecta drift)
        comparar_distribuciones_train_test(X, y, fin_train, fin_test, nombre)

    return caracteristicas


def graficar_resumen(caracteristicas: list):
    """
    Grafica ECE vs balance de clases y ECE vs volatilidad,
    para ver si hay correlación visual.
    """
    df_c = pd.DataFrame(caracteristicas)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(df_c['nombre'], df_c['ece'], color='steelblue')
    axes[0].set_title('ECE por ventana')
    axes[0].set_ylabel('Expected Calibration Error')
    axes[0].axhline(0.05, color='red', linestyle='--', label='Umbral problema')
    axes[0].legend()

    axes[1].bar(df_c['nombre'], df_c['balance'] * 100, color='orange')
    axes[1].set_title('% Target positivo por ventana')
    axes[1].set_ylabel('% positivos')
    axes[1].axhline(50, color='gray', linestyle='--')

    axes[2].bar(df_c['nombre'], df_c['atr_avg'], color='green')
    axes[2].set_title('Volatilidad (ATR relativo) por ventana')
    axes[2].set_ylabel('ATR relativo promedio')

    plt.tight_layout()
    plt.savefig('diagnostico_ventanas.png', dpi=150)
    plt.show()
    print("\n✅ Gráfico guardado como diagnostico_ventanas.png")


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔍 Diagnóstico de ventanas con mala calibración\n")

    df = pd.read_parquet(ARCHIVO_DATASET)
    X  = df[FEATURES]
    y  = df['target']

    print(f"Dataset: {len(df):,} filas")

    caracteristicas = evaluar_calibracion_por_ventana(X, y)

    print(f"\n\n{'='*60}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(f"  {'Ventana':<8} {'ECE':>8} {'Balance':>10} {'ADX avg':>10} {'ATR avg':>10}")
    for c in caracteristicas:
        print(f"  {c['nombre']:<8} {c['ece']:>8.4f} {c['balance']*100:>9.1f}% "
              f"{c['adx_avg']:>10.1f} {c['atr_avg']:>10.4f}")

    graficar_resumen(caracteristicas)