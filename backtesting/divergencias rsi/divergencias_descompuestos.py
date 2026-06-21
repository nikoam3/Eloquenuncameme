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
from divergencias import calcular_divergencias  # reusamos el cálculo
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
VENTANA_EXTREMO = 25  # la que mejor te funcionó

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


def walk_forward_auc_silencioso(X: pd.DataFrame, y: pd.Series) -> list:
    """
    Misma lógica de walk-forward, pero sin imprimir nada por
    ventana — solo devuelve la lista de AUCs. Útil para correr
    muchas configuraciones sin saturar la consola.
    """
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


def ver_coeficientes_modelo(X: pd.DataFrame, y: pd.Series, columnas: list):
    """
    Entrena un modelo con TODOS los datos (sin walk-forward,
    solo para inspeccionar coeficientes) y muestra qué peso le
    dio a cada feature. Como están estandarizadas (StandardScaler),
    los coeficientes son directamente comparables entre sí.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo.fit(X_scaled, y)

    coefs = pd.Series(modelo.coef_[0], index=columnas)
    coefs_abs = coefs.abs().sort_values(ascending=False)

    print(f"\n  Importancia de features (|coeficiente|, mayor = más peso):")
    print(f"  {'Feature':<28} {'Coeficiente':>12} {'Abs':>8}")
    print(f"  {'─'*50}")
    for feat in coefs_abs.index:
        valor = coefs[feat]
        marca = "🆕" if feat in FEATURES_DIVERGENCIA else "  "
        print(f"  {marca} {feat:<25} {valor:>+12.4f} {abs(valor):>8.4f}")


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Descomposición del aporte de cada feature de divergencia\n")

    print("Cargando datos...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    df_15m = df_15m.dropna()

    df_15m = calcular_divergencias(df_15m, ventana=VENTANA_EXTREMO)

    features_df, df_15m_completo = construir_features("ETHUSDT")
    target = crear_target(
        df_15m_completo.loc[features_df.index],
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, max_velas=MAX_VELAS
    )

    divergencias_alineadas = df_15m[FEATURES_DIVERGENCIA].reindex(
        features_df.index
    )
    df_ml = features_df.join(target).join(divergencias_alineadas).dropna()

    print(f"Dataset final: {len(df_ml):,} filas\n")

    y = df_ml['target']

    # ── Baseline: solo features base ────────────────────────
    print(f"{'='*60}")
    print("  PASO 1: Baseline (solo features base)")
    print(f"{'='*60}")
    X_base = df_ml[FEATURES_BASE]
    aucs_base = walk_forward_auc_silencioso(X_base, y)
    auc_base_prom = np.mean(aucs_base)
    print(f"  AUC promedio: {auc_base_prom:.4f}")

    # ── Cada feature de divergencia, SOLA, sumada al baseline ──
    print(f"\n{'='*60}")
    print("  PASO 2: Aporte INDIVIDUAL de cada feature nueva")
    print(f"{'='*60}")
    print(f"  {'Feature agregada':<28} {'AUC':>8} {'vs base':>10}")
    print(f"  {'─'*48}")

    resultados_individuales = {}
    for feat in FEATURES_DIVERGENCIA:
        cols = FEATURES_BASE + [feat]
        X_var = df_ml[cols]
        aucs_var = walk_forward_auc_silencioso(X_var, y)
        auc_var_prom = np.mean(aucs_var)
        dif = auc_var_prom - auc_base_prom
        resultados_individuales[feat] = auc_var_prom
        flag = "✅" if dif > 0.003 else ("➖" if dif > 0 else "⚠️")
        print(f"  +{feat:<27} {auc_var_prom:>8.4f} {dif:>+9.4f} {flag}")

    # ── Las tres juntas (referencia, lo que ya midieron antes) ──
    print(f"\n{'='*60}")
    print("  PASO 3: Las tres juntas (referencia)")
    print(f"{'='*60}")
    X_completo = df_ml[FEATURES_BASE + FEATURES_DIVERGENCIA]
    aucs_completo = walk_forward_auc_silencioso(X_completo, y)
    auc_completo_prom = np.mean(aucs_completo)
    print(f"  AUC promedio: {auc_completo_prom:.4f}  "
          f"(vs base: {auc_completo_prom - auc_base_prom:+.4f})")

    # ── Probar combinaciones de DOS features ────────────────
    print(f"\n{'='*60}")
    print("  PASO 4: Combinaciones de DOS features")
    print(f"{'='*60}")
    from itertools import combinations
    for combo in combinations(FEATURES_DIVERGENCIA, 2):
        cols = FEATURES_BASE + list(combo)
        X_var = df_ml[cols]
        aucs_var = walk_forward_auc_silencioso(X_var, y)
        auc_var_prom = np.mean(aucs_var)
        dif = auc_var_prom - auc_base_prom
        nombre_combo = " + ".join(combo)
        print(f"  +{nombre_combo:<45} {auc_var_prom:.4f} ({dif:+.4f})")

    # ── Coeficientes del modelo completo (las 3 + base) ─────
    print(f"\n{'='*60}")
    print("  PASO 5: Pesos asignados por el modelo (las 3 juntas)")
    print(f"{'='*60}")
    ver_coeficientes_modelo(
        X_completo, y, FEATURES_BASE + FEATURES_DIVERGENCIA
    )

    # ── Resumen final con recomendación ─────────────────────
    print(f"\n\n{'='*60}")
    print("  RESUMEN Y RECOMENDACIÓN")
    print(f"{'='*60}")
    mejor_individual = max(resultados_individuales,
                           key=resultados_individuales.get)
    print(f"  Baseline:              {auc_base_prom:.4f}")
    for feat, auc in resultados_individuales.items():
        print(f"  + {feat:<28}: {auc:.4f} ({auc-auc_base_prom:+.4f})")
    print(f"  + las 3 juntas:          {auc_completo_prom:.4f} "
          f"({auc_completo_prom-auc_base_prom:+.4f})")
    print(f"\n  Mejor feature individual: {mejor_individual}")

"""
PASO 1: Confirma el baseline (debería coincidir con tu corrida anterior)

PASO 2: Agrega cada feature SOLA al baseline
         → si una sola explica casi todo el +0.012 del total,
           las otras dos son redundantes

PASO 3: Las tres juntas (tu resultado anterior, de referencia)

PASO 4: Pares de features
         → ¿hay sinergia entre dos features que no se ve
           al mirarlas solas? A veces dos señales débiles
           se complementan

PASO 5: Coeficientes del modelo final
         → cuánto "peso" interno le asigna la regresión
           logística a cada una (con datos estandarizados,
           los coeficientes son comparables directamente)
"""