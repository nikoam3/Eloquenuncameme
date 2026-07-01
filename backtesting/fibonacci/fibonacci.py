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
VENTANA_PENDIENTE = 20
VENTANA_SWING     = 50  # velas hacia atrás para definir el swing

NIVELES_FIBO = [0.236, 0.382, 0.5, 0.618, 0.786]

# Baseline actualizado: incluye las DOS features ya confirmadas
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
# CÁLCULO DE NIVELES DE FIBONACCI
# ============================================================
def calcular_distancia_fibonacci(df: pd.DataFrame,
                                 ventana: int = VENTANA_SWING) -> pd.Series:
    """
    Para cada vela, define un swing usando el máximo y mínimo de
    las últimas `ventana` velas (sin incluir la actual), determina
    si es alcista o bajista según qué extremo ocurrió más reciente,
    calcula los 5 niveles de Fibonacci clásicos, y devuelve la
    distancia porcentual del precio actual al nivel MÁS CERCANO.

    Valor cercano a 0: el precio está justo sobre un nivel de
                       Fibonacci (zona "relevante" según la teoría)
    Valor alto: el precio está lejos de cualquier nivel
    """
    n = len(df)
    distancias = np.full(n, np.nan)

    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values

    for i in range(ventana, n):
        ventana_high = highs[i-ventana:i]
        ventana_low  = lows[i-ventana:i]

        idx_max = np.argmax(ventana_high)
        idx_min = np.argmin(ventana_low)

        maximo = ventana_high[idx_max]
        minimo = ventana_low[idx_min]
        rango  = maximo - minimo

        if rango == 0:
            continue  # sin movimiento, no hay niveles que calcular

        # Si el máximo ocurrió DESPUÉS del mínimo → swing alcista
        # (los niveles se miden retrocediendo desde el máximo)
        swing_alcista = idx_max > idx_min

        if swing_alcista:
            niveles = [maximo - rango * r for r in NIVELES_FIBO]
        else:
            niveles = [minimo + rango * r for r in NIVELES_FIBO]

        precio_actual = closes[i]
        distancia_a_niveles = [
            abs(precio_actual - nivel) / precio_actual * 100
            for nivel in niveles
        ]
        distancias[i] = min(distancia_a_niveles)

    return pd.Series(distancias, index=df.index)


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
    print("🔬 Evaluación de niveles de Fibonacci (Módulo 7B.3)\n")

    print("Cargando datos y calculando indicadores...")
    df_15m = cargar_velas("ETHUSDT", "15m")


    print(f"Calculando distancia a niveles de Fibonacci "
          f"(ventana swing={VENTANA_SWING})...")
    print("  (tiene loop, puede tardar unos segundos)")
    df_15m['distancia_fibonacci'] = calcular_distancia_fibonacci(df_15m)

    print("\nConstruyendo features base (multi-timeframe)...")
    features_df, df_15m_completo = construir_features("ETHUSDT")
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, max_velas=MAX_VELAS
    )

    cols_nuevas = ['distancia_fibonacci']
    nuevas_alineadas = df_15m[cols_nuevas].reindex(features_df.index)

    df_ml = features_df.join(target).join(nuevas_alineadas).dropna()
    print(f"Dataset final: {len(df_ml):,} filas\n")

    y = df_ml['target']

    # ── Baseline actualizado ─────────────────────────────────
    print(f"{'='*60}")
    print("  Baseline (incluye rsi_pendiente + macd_histograma)")
    print(f"{'='*60}")
    X_base = df_ml[FEATURES_BASE_ACTUALIZADO]
    aucs_base = walk_forward_auc_silencioso(X_base, y)
    auc_base_prom = np.mean(aucs_base)
    print(f"  AUC promedio: {auc_base_prom:.4f}")
    print(f"  Por ventana: {[round(a, 4) for a in aucs_base]}")

    # ── Con la nueva feature de Fibonacci ────────────────────
    print(f"\n{'='*60}")
    print("  Con distancia_fibonacci agregada")
    print(f"{'='*60}")
    cols = FEATURES_BASE_ACTUALIZADO + ['distancia_fibonacci']
    X_fibo = df_ml[cols]
    aucs_fibo = walk_forward_auc_silencioso(X_fibo, y)
    auc_fibo_prom = np.mean(aucs_fibo)
    std_fibo = np.std(aucs_fibo)
    dif = auc_fibo_prom - auc_base_prom

    print(f"  AUC promedio: {auc_fibo_prom:.4f} "
          f"(vs base: {dif:+.4f})")
    print(f"  Por ventana: {[round(a, 4) for a in aucs_fibo]}")
    print(f"  Desviación entre ventanas: {std_fibo:.4f}")

    # ── Chequeo de sanidad: ¿tiene sentido la teoría? ───────
    print(f"\n{'='*60}")
    print("  Chequeo de sanidad: tasa de éxito según cercanía")
    print(f"{'='*60}")
    df_ml['cerca_de_nivel'] = df_ml['distancia_fibonacci'] < df_ml['distancia_fibonacci'].median()
    target_cerca  = df_ml.loc[df_ml['cerca_de_nivel'], 'target'].mean()
    target_lejos  = df_ml.loc[~df_ml['cerca_de_nivel'], 'target'].mean()
    print(f"  Cerca de un nivel de Fibonacci: {target_cerca*100:.1f}% éxito")
    print(f"  Lejos de niveles de Fibonacci:  {target_lejos*100:.1f}% éxito")
    print(f"  (la teoría sugiere que cerca de un nivel debería haber")
    print(f"   más probabilidad de reacción del precio, sea a favor")
    print(f"   o en contra — no necesariamente más éxito de COMPRA)")

    print(f"\n\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    print(f"  Baseline:              {auc_base_prom:.4f}")
    print(f"  + distancia_fibonacci: {auc_fibo_prom:.4f} ({dif:+.4f})")
    if dif > 0.003:
        print(f"\n  ✅ Aporta señal real. Vale la pena explorar más")
        print(f"     (ej. probar otras ventanas de swing).")
    else:
        print(f"\n  ⚠️  No aporta señal distinguible del baseline.")
        print(f"     Consistente con el patrón general de este módulo:")
        print(f"     la mayoría de las ideas razonables no mejoran el AUC.")