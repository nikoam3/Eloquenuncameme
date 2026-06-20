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
# Reusamos las funciones de simulación tal cual están en backtesting_ml.py
from backtesting_ml import comparar, simular_ml, FEATURES

# ============================================================
# CONFIGURACIÓN
# ============================================================
PAR             = "ETHUSDT"
CAPITAL_INICIAL = 100.0
STOP_LOSS       = 0.025
TAKE_PROFIT     = 0.010
MAX_VELAS       = 96
UMBRAL_PROB     = 0.63   # mismo umbral que el modelo original, a propósito
COMISION       = 0.001


# ============================================================
# WALK FORWARD CON MODELO CALIBRADO
# ============================================================
def walk_forward_calibrado(features_df: pd.DataFrame,
                           df_15m: pd.DataFrame,
                           umbral_prob: float = UMBRAL_PROB) -> tuple:
    """
    Igual que walk_forward_backtesting() del módulo original,
    pero entrenando con CalibratedClassifierCV (Isotonic) en
    lugar de LogisticRegression directa.
    """
    n          = len(features_df)
    n_ventanas = 4
    tam_test   = n // (n_ventanas + 1)

    equity_total = []
    ops_total    = []

    print(f"\nWalk Forward Backtesting ML CALIBRADO (umbral={umbral_prob}):")
    print(f"{'─'*55}")

    for i in range(n_ventanas):
        fin_train = tam_test * (i + 1)
        fin_test  = tam_test * (i + 2)

        f_train = features_df.iloc[:fin_train]
        f_test  = features_df.iloc[fin_train:fin_test]
        d_test  = df_15m.loc[f_test.index]

        X_train  = f_train[FEATURES]
        y_train  = f_train['target']
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # ── ÚNICO CAMBIO: modelo calibrado en vez de directo ──
        modelo = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            method='isotonic', cv=3
        )
        modelo.fit(X_scaled, y_train)

        # Simulamos con datos futuros (misma función que siempre)
        ops, equity, cap = simular_ml(
            f_test, d_test, modelo, scaler, umbral_prob
        )

        retorno = (cap / CAPITAL_INICIAL - 1) * 100
        n_ops   = len(ops[ops['ganancia_pct'].notna()])
        fecha_i = f_test.index[0].strftime('%m/%Y')
        fecha_f = f_test.index[-1].strftime('%m/%Y')

        print(f"  V{i+1} {fecha_i}-{fecha_f}: "
              f"retorno {retorno:+.2f}%  ops: {n_ops}")

        if not equity.empty:
            equity_total.append(equity)
        if not ops.empty:
            ops_total.append(ops)

    equity_df = pd.concat(equity_total).drop_duplicates('time')
    ops_df    = (pd.concat(ops_total).drop_duplicates()
                 if ops_total else pd.DataFrame())
    cap_final = equity_df['equity'].iloc[-1]

    return ops_df, equity_df, cap_final


def barrer_umbrales(features_df: pd.DataFrame, df_15m: pd.DataFrame,
                     umbrales: list) -> pd.DataFrame:
    """
    Corre el walk-forward calibrado para varios umbrales y
    devuelve una tabla comparativa. Permite elegir el umbral
    que mejor balancea retorno, win rate y frecuencia.
    """
    resultados = []
    for u in umbrales:
        print(f"\n{'#'*55}")
        print(f"  Probando UMBRAL_PROB = {u}")
        print(f"{'#'*55}")

        ops, equity, cap = walk_forward_calibrado(features_df, df_15m, u)

        if ops.empty:
            resultados.append({
                'umbral': u, 'retorno_pct': 0, 'n_ops': 0,
                'win_rate': 0, 'max_dd': 0
            })
            continue

        cerradas = ops[ops['ganancia_pct'].notna()]
        retorno  = (cap / CAPITAL_INICIAL - 1) * 100
        n_ops    = len(cerradas)
        win_rate = (len(cerradas[cerradas['ganancia_pct'] > 0])
                    / n_ops * 100) if n_ops > 0 else 0

        vals     = equity['equity'].values
        peak     = np.maximum.accumulate(vals)
        dd       = (vals - peak) / peak * 100
        max_dd   = dd.min()

        resultados.append({
            'umbral': u,
            'retorno_pct': retorno,
            'n_ops': n_ops,
            'win_rate': win_rate,
            'max_dd': max_dd
        })

    return pd.DataFrame(resultados)


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("📊 Backtesting con modelo CALIBRADO (Isotonic)\n")

    print("Cargando datos...")
    df_15m = cargar_velas("ETHUSDT", "15m")
    df_15m['MM']  = ta.ema(df_15m['Close'], length=200)
    df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
    adx = ta.adx(df_15m['High'], df_15m['Low'],
                 df_15m['Close'], length=14)
    df_15m['ADX'] = adx['ADX_14']
    df_15m = df_15m.dropna()

    print("Construyendo features ML...")
    features_df, _ = construir_features("ETHUSDT")
    target = crear_target(df_15m.loc[features_df.index],
                          stop_loss=STOP_LOSS,
                          take_profit=TAKE_PROFIT,
                          max_velas=MAX_VELAS)
    features_df = features_df.join(target).dropna()

    # ── PASO 1: backtest con el mismo umbral 0.63 ───────────
    print(f"\n{'='*55}")
    print(f"  PASO 1: Backtest con umbral original (0.63)")
    print(f"{'='*55}")

    ops_cal, equity_cal, cap_cal = walk_forward_calibrado(
        features_df, df_15m, UMBRAL_PROB
    )

    print(f"\n{'='*50}")
    print("   RESULTADOS — MODELO CALIBRADO (umbral 0.63)")
    print(f"{'='*50}")
    comparar("ML Calibrado (Isotonic, umbral=0.63)",
             ops_cal, equity_cal, cap_cal)

    # ── PASO 2: barrido de umbrales ─────────────────────────
    print(f"\n\n{'='*55}")
    print(f"  PASO 2: Barrido de umbrales (modelo calibrado)")
    print(f"{'='*55}")

    #umbrales_a_probar = [0.55, 0.60, 0.63, 0.65, 0.70, 0.75]
    umbrales_a_probar = [0.75, 0.78, 0.80, 0.83, 0.85]
    tabla = barrer_umbrales(features_df, df_15m, umbrales_a_probar)

    print(f"\n\n{'='*60}")
    print("  TABLA COMPARATIVA DE UMBRALES (modelo calibrado)")
    print(f"{'='*60}")
    print(f"  {'Umbral':>8} {'Retorno':>10} {'Ops':>6} "
          f"{'WinRate':>9} {'MaxDD':>9}")
    print(f"  {'─'*45}")
    for _, r in tabla.iterrows():
        print(f"  {r['umbral']:>8.2f} {r['retorno_pct']:>9.2f}% "
              f"{r['n_ops']:>6.0f} {r['win_rate']:>8.1f}% "
              f"{r['max_dd']:>8.2f}%")

    print(f"\n  Referencia — modelo SIN calibrar (de tu backtest original):")
    print(f"  {'umbral 0.63':>8}: retorno +24.63%  ops: 853  "
          f"win rate: 77.0%  maxDD: -29.73%")
    
