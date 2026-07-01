"""
Paso 1: entrenar el modelo normal (como siempre) en cada ventana
Paso 2: en el test, separar las predicciones en terciles de ATR
         (bajo / medio / alto, dentro de esa ventana)
Paso 3: para cada tercil, medir la precisión real en distintos
         umbrales (0.70, 0.75, 0.80, 0.85)
Paso 4: ver si el umbral "óptimo" (mejor balance precisión/volumen)
         se mueve entre terciles
"""
import sys
import os
# Encuentra la carpeta principal 'backtesting' subiendo un nivel
ruta_principal = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ruta_principal)
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from features import construir_features, crear_target
from database import cargar_velas
import pandas_ta as ta
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
UMBRALES_PROBAR = [0.70, 0.75, 0.80, 0.85]

FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200', 'ema9_vs_ema26',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'rsi_pendiente_vs_precio', 'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


def evaluar_ventana_por_volatilidad(X: pd.DataFrame, y: pd.Series,
                                    fin_train: int, fin_test: int,
                                    nombre: str) -> pd.DataFrame:
    """
    Entrena el modelo calibrado (igual que producción) en una ventana
    de walk-forward, y mide la precisión real por tercil de
    volatilidad (atr_relativo) dentro del set de test.
    """
    X_train = X.iloc[:fin_train]
    y_train = y.iloc[:fin_train]
    X_test  = X.iloc[fin_train:fin_test]
    y_test  = y.iloc[fin_train:fin_test]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    modelo = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        method='isotonic', cv=3
    )
    modelo.fit(X_tr_sc, y_train)
    probs = modelo.predict_proba(X_te_sc)[:, 1]

    # Terciles de volatilidad DENTRO de esta ventana de test
    atr_test = X_test['atr_relativo'].values
    terciles = pd.qcut(atr_test, q=3, labels=['Baja', 'Media', 'Alta'])

    resultados = []
    for tercil in ['Baja', 'Media', 'Alta']:
        mask_tercil = (terciles == tercil)
        y_tercil = y_test.values[mask_tercil]
        probs_tercil = probs[mask_tercil]

        for umbral in UMBRALES_PROBAR:
            mask_umbral = probs_tercil >= umbral
            n_señales = mask_umbral.sum()
            precision = y_tercil[mask_umbral].mean() if n_señales > 0 else np.nan

            resultados.append({
                'ventana': nombre,
                'volatilidad': tercil,
                'umbral': umbral,
                'n_señales': n_señales,
                'precision_real': precision,
                'pct_del_tercil': n_señales / mask_tercil.sum() * 100
            })

    return pd.DataFrame(resultados)


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Diagnóstico: ¿el umbral óptimo varía con la volatilidad?\n")

    df = pd.read_parquet('dataset_ml.parquet')
    X  = df[FEATURES]
    y  = df['target']

    print(f"Dataset: {len(df):,} filas")

    n        = len(X)
    tam_test = n // (N_VENTANAS + 1)

    todos_resultados = []
    for i in range(N_VENTANAS):
        fin_train = tam_test * (i + 1)
        fin_test  = tam_test * (i + 2)
        print(f"\nProcesando V{i+1}...")
        r = evaluar_ventana_por_volatilidad(
            X, y, fin_train, fin_test, f"V{i+1}"
        )
        todos_resultados.append(r)

    df_resultados = pd.concat(todos_resultados, ignore_index=True)

    # ── Tabla detallada por ventana ─────────────────────────
    print(f"\n\n{'='*75}")
    print("  DETALLE POR VENTANA Y TERCIL DE VOLATILIDAD")
    print(f"{'='*75}")
    for ventana in ['V1', 'V2', 'V3', 'V4']:
        print(f"\n  {ventana}:")
        print(f"  {'Volatilidad':<10} {'Umbral':>8} {'N señales':>10} "
              f"{'Precisión real':>15} {'% del tercil':>13}")
        print(f"  {'─'*60}")
        subset = df_resultados[df_resultados['ventana'] == ventana]
        for _, row in subset.iterrows():
            prec_str = (f"{row['precision_real']*100:.1f}%"
                       if not pd.isna(row['precision_real']) else "N/A")
            print(f"  {row['volatilidad']:<10} {row['umbral']:>8.2f} "
                  f"{row['n_señales']:>10.0f} {prec_str:>15} "
                  f"{row['pct_del_tercil']:>12.1f}%")

    # ── Resumen agregado: promedio entre las 4 ventanas ─────
    print(f"\n\n{'='*75}")
    print("  RESUMEN: precisión real promedio (4 ventanas) por")
    print("  tercil de volatilidad y umbral")
    print(f"{'='*75}")

    resumen = df_resultados.groupby(['volatilidad', 'umbral']).agg(
        precision_promedio=('precision_real', 'mean'),
        n_señales_total=('n_señales', 'sum')
    ).reset_index()

    print(f"\n  {'Volatilidad':<10} {'Umbral':>8} {'Precisión prom.':>16} "
          f"{'N señales total':>16}")
    print(f"  {'─'*55}")
    for _, row in resumen.iterrows():
        prec_str = (f"{row['precision_promedio']*100:.1f}%"
                   if not pd.isna(row['precision_promedio']) else "N/A")
        print(f"  {row['volatilidad']:<10} {row['umbral']:>8.2f} "
              f"{prec_str:>16} {row['n_señales_total']:>16.0f}")

    # ── Pregunta clave: ¿el umbral óptimo se mueve? ─────────
    print(f"\n\n{'='*75}")
    print("  PREGUNTA CLAVE: ¿la precisión a umbral fijo varía mucho")
    print("  entre niveles de volatilidad?")
    print(f"{'='*75}")
    for umbral in UMBRALES_PROBAR:
        subset = resumen[resumen['umbral'] == umbral]
        precisiones = subset['precision_promedio'].dropna()
        if len(precisiones) >= 2:
            spread = precisiones.max() - precisiones.min()
            print(f"\n  Umbral {umbral}: spread entre terciles = "
                  f"{spread*100:.1f} puntos porcentuales")
            for _, row in subset.iterrows():
                if not pd.isna(row['precision_promedio']):
                    print(f"    {row['volatilidad']:<8}: "
                          f"{row['precision_promedio']*100:.1f}%")

    print(f"\n  Interpretación:")
    print(f"  • Si el spread es chico (<5pp) en todos los umbrales:")
    print(f"    el umbral fijo actual ya es razonable, NO hace falta")
    print(f"    un mecanismo dinámico.")
    print(f"  • Si el spread es grande (>10pp) y consistente:")
    print(f"    hay evidencia real de que conviene diferenciar el")
    print(f"    umbral según volatilidad.")

"""
Si la precisión real a umbral=0.80 es:
  Baja volatilidad:  85%
  Media volatilidad: 83%
  Alta volatilidad:  82%
  → spread chico (3pp) → el umbral fijo está bien, no hace
    falta complicar el sistema con un mecanismo dinámico

Si en cambio es:
  Baja volatilidad:  90%
  Media volatilidad: 80%
  Alta volatilidad:  65%
  → spread grande (25pp) → hay una señal real de que el
    mismo umbral "significa" cosas muy distintas según
    contexto, y ahí sí vale la pena construir 7C completo
"""