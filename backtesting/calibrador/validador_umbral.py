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
from backtesting_ml import simular_ml, FEATURES
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
PAR             = "ETHUSDT"
CAPITAL_INICIAL = 100.0
STOP_LOSS       = 0.025
TAKE_PROFIT     = 0.010
MAX_VELAS       = 96


def entrenar_y_simular_ventana(features_df, df_15m,
                               fin_train, fin_test, umbral_prob):
    """
    Entrena el modelo calibrado solo con datos hasta fin_train,
    y simula en la ventana [fin_train:fin_test].
    Es el mismo bloque que usábamos en el walk-forward, pero
    aislado para poder llamarlo ventana por ventana.
    """
    f_train = features_df.iloc[:fin_train]
    f_test  = features_df.iloc[fin_train:fin_test]
    d_test  = df_15m.loc[f_test.index]
    X_train  = f_train[FEATURES]
    y_train  = f_train['target']
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    modelo = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        method='isotonic', cv=3
    )
    modelo.fit(X_scaled, y_train)

    ops, equity, cap = simular_ml(
        f_test, d_test, modelo, scaler, umbral_prob=umbral_prob
    )
    return ops, equity, cap


def metricas_de(ops: pd.DataFrame, equity: pd.DataFrame, cap: float) -> dict:
    """Extrae métricas resumen de una simulación."""
    if ops.empty:
        return {'retorno_pct': 0, 'n_ops': 0, 'win_rate': 0, 'max_dd': 0}

    cerradas = ops[ops['ganancia_pct'].notna()]
    retorno  = (cap / CAPITAL_INICIAL - 1) * 100
    n_ops    = len(cerradas)
    win_rate = (len(cerradas[cerradas['ganancia_pct'] > 0])
                / n_ops * 100) if n_ops > 0 else 0

    if not equity.empty:
        vals   = equity['equity'].values
        peak   = np.maximum.accumulate(vals)
        dd     = (vals - peak) / peak * 100
        max_dd = dd.min()
    else:
        max_dd = 0

    return {
        'retorno_pct': retorno,
        'n_ops': n_ops,
        'win_rate': win_rate,
        'max_dd': max_dd
    }


def paso_a_elegir_umbral(features_df, df_15m, umbrales: list,
                         n_ventanas_eleccion: int = 3) -> float:
    """
    PASO A: Recorre las primeras `n_ventanas_eleccion` ventanas
    (V1, V2, V3) y elige el umbral con mejor retorno PROMEDIO
    en esas ventanas únicamente. V4 no se toca acá.
    """
    n          = len(features_df)
    n_total    = 4  # mismas 4 ventanas de siempre
    tam_test   = n // (n_total + 1)

    print(f"\n{'='*60}")
    print(f"  PASO A — Elección de umbral usando V1, V2, V3")
    print(f"  (V4 queda reservada, no se mira en esta etapa)")
    print(f"{'='*60}")

    resultados_por_umbral = {u: [] for u in umbrales}

    for i in range(n_ventanas_eleccion):
        fin_train = tam_test * (i + 1)
        fin_test  = tam_test * (i + 2)
        fecha_i   = features_df.index[fin_train].strftime('%m/%Y')

        print(f"\n  Ventana V{i+1} ({fecha_i}):")
        for u in umbrales:
            ops, equity, cap = entrenar_y_simular_ventana(
                features_df, df_15m, fin_train, fin_test, u
            )
            m = metricas_de(ops, equity, cap)
            resultados_por_umbral[u].append(m['retorno_pct'])
            print(f"    umbral={u:.2f}  →  retorno {m['retorno_pct']:+.2f}%  "
                  f"({m['n_ops']:.0f} ops, WR={m['win_rate']:.1f}%)")

    print(f"\n  {'─'*45}")
    print(f"  Promedio de retorno (solo V1+V2+V3):")
    print(f"  {'Umbral':>8} {'Retorno promedio':>18}")
    for u in umbrales:
        promedio = np.mean(resultados_por_umbral[u])
        print(f"  {u:>8.2f} {promedio:>17.2f}%")

    mejor_umbral = max(resultados_por_umbral,
                       key=lambda u: np.mean(resultados_por_umbral[u]))
    print(f"\n  → Umbral elegido (mejor en V1+V2+V3): {mejor_umbral}")

    return mejor_umbral


def paso_b_validar_en_v4(features_df, df_15m, umbral_elegido: float):
    """
    PASO B: Usando el umbral ya decidido (sin tocarlo más),
    medimos qué tan bien funciona en V4 — la ventana que
    NUNCA participó en la elección del umbral.
    """
    n        = len(features_df)
    tam_test = n // (4 + 1)

    fin_train = tam_test * 4   # train = V1+V2+V3 completas
    fin_test  = tam_test * 5   # test  = V4

    fecha_i = features_df.index[fin_train].strftime('%m/%Y')
    fecha_f = features_df.index[fin_test - 1].strftime('%m/%Y')

    print(f"\n{'='*60}")
    print(f"  PASO B — Validación HONESTA en V4 ({fecha_i}-{fecha_f})")
    print(f"  Umbral usado: {umbral_elegido} (decidido SIN ver V4)")
    print(f"{'='*60}")

    ops, equity, cap = entrenar_y_simular_ventana(
        features_df, df_15m, fin_train, fin_test, umbral_elegido
    )
    m = metricas_de(ops, equity, cap)

    print(f"\n  Resultado en V4 (datos nunca vistos para elegir umbral):")
    print(f"    Retorno:     {m['retorno_pct']:+.2f}%")
    print(f"    Operaciones: {m['n_ops']:.0f}")
    print(f"    Win Rate:    {m['win_rate']:.1f}%")
    print(f"    Max DD:      {m['max_dd']:.2f}%")

    return m


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Validación honesta del umbral elegido\n")

    print("Cargando datos...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['MM']  = ta.ema(df_15m['Close'], length=200)
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    adx = ta.adx(df_15m['High'], df_15m['Low'],
                 df_15m['Close'], length=14)
    df_15m['ADX'] = adx['ADX_14']
    df_15m['MACD_hist'] = ta.macd(df_15m['Close'], fast=12, slow=26, signal=6)['MACDh_12_26_6']

    df_15m = df_15m.dropna()

    print("Construyendo features ML...")
    features_df, _ = construir_features("ETHUSDT")

    target = crear_target(_.loc[features_df.index],
                          stop_loss=STOP_LOSS,
                          take_profit=TAKE_PROFIT,
                          max_velas=MAX_VELAS)
    features_df = features_df.join(target).dropna()
    
    umbrales_candidatos = [0.7, 0.75, 0.78, 0.80, 0.83, 0.85]

    # PASO A: elegir con V1+V2+V3
    umbral_elegido = paso_a_elegir_umbral(
        features_df, _, umbrales_candidatos,
        n_ventanas_eleccion=3
    )

    # PASO B: validar con V4 (nunca visto antes)
    resultado_v4 = paso_b_validar_en_v4(
        features_df, _, umbral_elegido
    )

    print(f"\n\n{'='*60}")
    print("  CONCLUSIÓN")
    print(f"{'='*60}")
    print(f"  Umbral elegido con V1-V3: {umbral_elegido}")
    print(f"  Resultado en V4 (honesto): "
          f"{resultado_v4['retorno_pct']:+.2f}% "
          f"({resultado_v4['n_ops']:.0f} ops, "
          f"WR={resultado_v4['win_rate']:.1f}%)")

    if resultado_v4['retorno_pct'] > 0 and resultado_v4['n_ops'] > 30:
        print(f"\n  ✅ El umbral generaliza razonablemente bien a datos nuevos.")
    elif resultado_v4['n_ops'] <= 30:
        print(f"\n  ⚠️  Pocas operaciones en V4 — resultado poco confiable, "
              f"no se puede concluir con certeza.")
    else:
        print(f"\n  ⚠️  El umbral NO generalizó bien — revisar antes de "
              f"llevar a producción.")
        
"""
PASO A:
  Para cada umbral candidato [0.75, 0.78, 0.80, 0.83, 0.85]:
    Entrena y simula en V1, V2, V3 (cada una con su propio
    walk-forward, igual que antes)
    Promedia el retorno de esas 3 ventanas
  Elige el umbral con mejor promedio
  
  → V4 todavía no fue tocado en absoluto

PASO B:
  Toma el umbral ganador de PASO A
  Entrena con V1+V2+V3 completas como "pasado"
  Simula en V4 como si fuera el futuro real
  Reporta el resultado SIN posibilidad de haber hecho trampa
"""