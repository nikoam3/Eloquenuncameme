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
VENTANA_EXTREMO = 25  # velas hacia atrás para buscar max/min local


FEATURES_BASE = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]

FEATURES_DIVERGENCIA = [
    'divergencia_bajista_rsi',
    'divergencia_alcista_rsi',
    'rsi_pendiente_vs_precio'
]


# ============================================================
# CÁLCULO DE DIVERGENCIAS
# ============================================================
def calcular_divergencias(df: pd.DataFrame,
                          ventana: int = VENTANA_EXTREMO) -> pd.DataFrame:
    """
    Detecta divergencias entre precio y RSI usando una ventana
    deslizante de `ventana` velas hacia atrás.

    Lógica:
      - Comparamos el High actual contra el máximo de High de
        las últimas `ventana` velas (sin contar la actual).
      - Comparamos el RSI actual contra el máximo de RSI del
        mismo período.
      - Divergencia bajista: precio supera su máximo reciente,
        pero el RSI NO supera el suyo (momentum más débil).
      - Análogo para mínimos = divergencia alcista.

    rsi_pendiente_vs_precio es la versión continua: compara
    cuánto cambió el precio (%) vs cuánto cambió el RSI en el
    mismo período, normalizado. Si el precio sube fuerte y el
    RSI casi no se mueve, el valor es alto (divergencia fuerte).
    """
    df = df.copy()

    # Máximos/mínimos móviles de las `ventana` velas PREVIAS
    # (shift(1) para no incluir la vela actual en la comparación)
    max_high_previo = df['High'].rolling(ventana).max().shift(1)
    min_low_previo   = df['Low'].rolling(ventana).min().shift(1)
    max_rsi_previo   = df['RSI'].rolling(ventana).max().shift(1)
    min_rsi_previo   = df['RSI'].rolling(ventana).min().shift(1)

    # Divergencia bajista: nuevo máximo de precio, RSI no acompaña
    nuevo_max_precio = df['High'] > max_high_previo
    rsi_no_acompaña_max = df['RSI'] < max_rsi_previo
    df['divergencia_bajista_rsi'] = (
        nuevo_max_precio & rsi_no_acompaña_max
    ).astype(int)

    # Divergencia alcista: nuevo mínimo de precio, RSI no acompaña
    nuevo_min_precio = df['Low'] < min_low_previo
    rsi_no_acompaña_min = df['RSI'] > min_rsi_previo
    df['divergencia_alcista_rsi'] = (
        nuevo_min_precio & rsi_no_acompaña_min
    ).astype(int)

    # Versión continua: cambio % de precio vs cambio de RSI
    # en la ventana, normalizado por la volatilidad del RSI
    cambio_precio_pct = (
        df['Close'] / df['Close'].shift(ventana) - 1
    ) * 100
    cambio_rsi = df['RSI'] - df['RSI'].shift(ventana)

    # Si el precio sube mucho y el RSI sube poco (o baja),
    # el ratio se vuelve alto/negativo => señal de divergencia
    # Usamos un epsilon para evitar división por cero
    df['rsi_pendiente_vs_precio'] = (
        cambio_precio_pct - cambio_rsi
    )

    return df


def analizar_frecuencia_divergencias(df: pd.DataFrame):
    """Muestra qué tan seguido aparecen las divergencias detectadas."""
    n = len(df)
    n_bajista = df['divergencia_bajista_rsi'].sum()
    n_alcista = df['divergencia_alcista_rsi'].sum()

    print(f"\n  Frecuencia de divergencias detectadas:")
    print(f"    Divergencia bajista: {n_bajista:,} velas "
          f"({n_bajista/n*100:.2f}%)")
    print(f"    Divergencia alcista: {n_alcista:,} velas "
          f"({n_alcista/n*100:.2f}%)")

    # ¿Las divergencias coinciden con mejor/peor target?
    if 'target' in df.columns:
        target_normal = df.loc[
            (df['divergencia_bajista_rsi'] == 0) &
            (df['divergencia_alcista_rsi'] == 0), 'target'
        ].mean()
        target_bajista = df.loc[
            df['divergencia_bajista_rsi'] == 1, 'target'
        ].mean()
        target_alcista = df.loc[
            df['divergencia_alcista_rsi'] == 1, 'target'
        ].mean()

        print(f"\n  Tasa de éxito del target según contexto:")
        print(f"    Sin divergencia:        {target_normal*100:.1f}%")
        print(f"    Con divergencia bajista: {target_bajista*100:.1f}%  "
              f"(esperaríamos MENOR éxito si la teoría es correcta)")
        print(f"    Con divergencia alcista: {target_alcista*100:.1f}%  "
              f"(esperaríamos MAYOR éxito si la teoría es correcta)")


# ============================================================
# COMPARACIÓN DE AUC: CON vs SIN DIVERGENCIAS
# ============================================================
def walk_forward_auc(X: pd.DataFrame, y: pd.Series, nombre: str) -> list:
    """
    Igual estructura que el walk-forward de 7A, pero esta vez
    medimos AUC en vez de ECE - es la métrica que nos interesa
    para saber si el modelo discrimina mejor.
    """
    n        = len(X)
    tam_test = n // (N_VENTANAS + 1)
    aucs     = []

    print(f"\n  {nombre}:")
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

        auc = roc_auc_score(y_test, probs)
        aucs.append(auc)
        fecha = X.index[fin_train].strftime('%m/%Y')
        print(f"    V{i+1} ({fecha}): AUC = {auc:.4f}")

    promedio = np.mean(aucs)
    print(f"    {'─'*30}")
    print(f"    Promedio AUC: {promedio:.4f}")
    return aucs


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Evaluación de divergencias RSI (Módulo 7B.1)\n")

    print("Cargando datos y calculando indicadores base...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    df_15m = df_15m.dropna()

    print("Calculando divergencias...")
    df_15m = calcular_divergencias(df_15m)

    print("Construyendo features base (multi-timeframe)...")
    features_df, df_15m_completo = construir_features("ETHUSDT")

    # Creamos el target con la misma lógica de siempre
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        max_velas=MAX_VELAS
    )

    # Unimos las divergencias (calculadas sobre df_15m) con el
    # dataframe de features, alineando por índice de tiempo
    divergencias_alineadas = df_15m[FEATURES_DIVERGENCIA].reindex(
        features_df.index
    )

    df_ml = features_df.join(target).join(divergencias_alineadas).dropna()

    print(f"\nDataset final: {len(df_ml):,} filas")

    analizar_frecuencia_divergencias(df_ml)

    # ── Comparación A: solo features base ───────────────────
    print(f"\n{'='*55}")
    print("  COMPARACIÓN DE AUC: base vs base+divergencias")
    print(f"{'='*55}")

    X_base = df_ml[FEATURES_BASE]
    y      = df_ml['target']
    aucs_base = walk_forward_auc(X_base, y, "SOLO features base")

    # ── Comparación B: features base + divergencias ────────
    X_completo = df_ml[FEATURES_BASE + FEATURES_DIVERGENCIA]
    aucs_completo = walk_forward_auc(
        X_completo, y, "Features base + DIVERGENCIAS"
    )

    # ── Resumen final ────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'='*55}")
    print(f"  {'Ventana':<10} {'Solo base':>12} {'Con divergencias':>18} "
          f"{'Diferencia':>12}")
    print(f"  {'─'*55}")
    for i in range(N_VENTANAS):
        dif = aucs_completo[i] - aucs_base[i]
        flag = "✅" if dif > 0.005 else ("⚠️" if dif < -0.005 else "➖")
        print(f"  V{i+1:<9} {aucs_base[i]:>12.4f} {aucs_completo[i]:>18.4f} "
              f"{dif:>+11.4f} {flag}")

    dif_promedio = np.mean(aucs_completo) - np.mean(aucs_base)
    print(f"  {'─'*55}")
    print(f"  {'Promedio':<10} {np.mean(aucs_base):>12.4f} "
          f"{np.mean(aucs_completo):>18.4f} {dif_promedio:>+11.4f}")

    print(f"\n  Interpretación:")
    if dif_promedio > 0.01:
        print(f"  ✅ Las divergencias aportan señal real. Vale la pena")
        print(f"     integrarlas al dataset de producción.")
    elif dif_promedio > 0.003:
        print(f"  ➖ Mejora marginal. Podría valer la pena, pero es")
        print(f"     un aporte pequeño — evaluar si vale la complejidad.")
    else:
        print(f"  ⚠️  Sin mejora significativa. Las divergencias, tal")
        print(f"     como las calculamos, no agregan información nueva")
        print(f"     que el modelo no tuviera ya.")

"""
1. Frecuencia de divergencias
   → ¿son raras o frecuentes? Si aparecen en <1% de las velas,
     aunque funcionen, su impacto en el AUC global será chico

2. Tasa de éxito según contexto
   → Validación de la TEORÍA antes de mirar el modelo:
     ¿con divergencia bajista el target realmente es peor?
     Si esto no se cumple, la feature no tiene sentido aunque
     suba el AUC por casualidad estadística

3. AUC walk-forward: base vs base+divergencias
   → La medición final que decide si la integramos
"""