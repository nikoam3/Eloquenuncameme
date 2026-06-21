import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_DATASET = 'dataset_ml.parquet'

# Velas de 15m: ~96 por día, ~2880 por mes (aproximado)
VELAS_POR_MES = 96 * 30

# Reservamos los últimos 3 meses como evaluación FIJA
MESES_EVALUACION = 3

# Cantidades de historia a probar, en meses, ANTES del corte de test
MESES_ENTRENAMIENTO_A_PROBAR = [3, 6, 9, 12, 15, 18, 21, 24]

FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'rsi_pendiente_vs_precio', 'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


def entrenar_y_evaluar(df: pd.DataFrame, meses_entrenamiento: int,
                       n_test: int) -> dict:
    """
    Entrena con los últimos `meses_entrenamiento` meses ANTES del
    corte de test, y evalúa siempre sobre el MISMO período de test
    (los últimos n_test velas del dataset).

    Esto aísla la variable que nos interesa: cuánta historia usar
    para entrenar, manteniendo fijo el "examen" sobre el que se mide.
    """
    n = len(df)
    inicio_test = n - n_test
    velas_entrenamiento = meses_entrenamiento * VELAS_POR_MES

    inicio_train = max(0, inicio_test - velas_entrenamiento)

    if inicio_train >= inicio_test:
        return None  # no hay suficiente historia para este caso

    df_train = df.iloc[inicio_train:inicio_test]
    df_test  = df.iloc[inicio_test:]

    X_train = df_train[FEATURES]
    y_train = df_train['target']
    X_test  = df_test[FEATURES]
    y_test  = df_test['target']

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    modelo = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        method='isotonic', cv=3
    )
    modelo.fit(X_tr_sc, y_train)
    probs = modelo.predict_proba(X_te_sc)[:, 1]

    auc = roc_auc_score(y_test, probs)

    return {
        'meses_entrenamiento': meses_entrenamiento,
        'n_filas_train': len(df_train),
        'fecha_inicio_train': df_train.index[0],
        'fecha_fin_train': df_train.index[-1],
        'auc': auc
    }


def evaluar_precision_real(df: pd.DataFrame, meses_entrenamiento: int,
                           n_test: int, umbral: float = 0.80) -> dict:
    """
    Además del AUC, medimos la precisión real con el umbral de
    producción (0.80), para ver el impacto práctico, no solo
    la métrica abstracta.
    """
    n = len(df)
    inicio_test = n - n_test
    velas_entrenamiento = meses_entrenamiento * VELAS_POR_MES
    inicio_train = max(0, inicio_test - velas_entrenamiento)

    if inicio_train >= inicio_test:
        return None

    df_train = df.iloc[inicio_train:inicio_test]
    df_test  = df.iloc[inicio_test:]

    X_train = df_train[FEATURES]
    y_train = df_train['target']
    X_test  = df_test[FEATURES]
    y_test  = df_test['target']

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    modelo = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        method='isotonic', cv=3
    )
    modelo.fit(X_tr_sc, y_train)
    probs = modelo.predict_proba(X_te_sc)[:, 1]

    mask = probs >= umbral
    n_señales = mask.sum()
    precision = y_test.values[mask].mean() if n_señales > 0 else np.nan

    return {
        'n_señales': n_señales,
        'precision_real': precision
    }


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("🔬 Experimento: ¿cuánta historia conviene para entrenar?\n")

    df = pd.read_parquet(ARCHIVO_DATASET)
    print(f"Dataset completo: {len(df):,} filas")
    print(f"Período: {df.index[0]} → {df.index[-1]}")

    n_test = MESES_EVALUACION * VELAS_POR_MES
    print(f"\nPeríodo de evaluación FIJO (siempre el mismo): "
          f"últimas {n_test:,} velas (~{MESES_EVALUACION} meses)")
    print(f"  Desde: {df.index[len(df) - n_test]}")
    print(f"  Hasta: {df.index[-1]}")

    print(f"\n{'='*70}")
    print("  RESULTADOS: AUC según cantidad de historia de entrenamiento")
    print(f"{'='*70}")
    print(f"  {'Meses train':>12} {'N filas':>10} {'Desde':>12} "
          f"{'AUC':>8} {'Precisión@0.80':>15} {'N señales':>10}")
    print(f"  {'─'*70}")

    resultados = []
    for meses in MESES_ENTRENAMIENTO_A_PROBAR:
        r_auc = entrenar_y_evaluar(df, meses, n_test)
        if r_auc is None:
            print(f"  {meses:>12}   (no hay suficiente historia disponible)")
            continue

        r_prec = evaluar_precision_real(df, meses, n_test)

        resultados.append({**r_auc, **r_prec})

        fecha_desde = r_auc['fecha_inicio_train'].strftime('%Y-%m')
        prec_str = (f"{r_prec['precision_real']*100:.1f}%"
                   if not pd.isna(r_prec['precision_real']) else "N/A")

        print(f"  {meses:>12} {r_auc['n_filas_train']:>10,} "
              f"{fecha_desde:>12} {r_auc['auc']:>8.4f} "
              f"{prec_str:>15} {r_prec['n_señales']:>10,.0f}")

    # ── Análisis de tendencia ────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  ANÁLISIS DE TENDENCIA")
    print(f"{'='*70}")

    df_res = pd.DataFrame(resultados)
    mejor_auc = df_res.loc[df_res['auc'].idxmax()]
    print(f"\n  Mejor AUC: {mejor_auc['auc']:.4f} "
          f"con {mejor_auc['meses_entrenamiento']:.0f} meses de entrenamiento")

    # Correlación simple: ¿más meses tiende a mejor AUC?
    correlacion = df_res['meses_entrenamiento'].corr(df_res['auc'])
    print(f"\n  Correlación (meses de entrenamiento vs AUC): {correlacion:+.3f}")
    print(f"  (cercano a +1: más historia ayuda consistentemente)")
    print(f"  (cercano a -1: menos historia ayuda consistentemente)")
    print(f"  (cercano a 0: no hay relación clara, el AUC no depende")
    print(f"   mucho de cuánta historia se use)")

    print(f"\n  Interpretación:")
    if correlacion > 0.5:
        print(f"  ✅ Hay evidencia de que MÁS historia ayuda. Considerar")
        print(f"     ampliar la base de datos histórica si es posible.")
    elif correlacion < -0.5:
        print(f"  ⚠️  Hay evidencia de que MENOS historia ayuda — el")
        print(f"     mercado pudo haber cambiado de régimen, y datos")
        print(f"     muy viejos podrían estar 'ensuciando' el aprendizaje.")
    else:
        print(f"  ➖ No hay una relación clara. La cantidad de historia,")
        print(f"     dentro del rango de 3-24 meses, no parece ser el")
        print(f"     factor más determinante para este modelo.")