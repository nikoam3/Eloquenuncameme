import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_DATASET = 'dataset_ml.parquet'
UMBRAL_PROB     = 0.60   # solo operamos si probabilidad >= 60%
N_VENTANAS      = 4      # ventanas de Walk Forward


# ============================================================
# CARGA DE DATOS
# ============================================================
def cargar_dataset() -> tuple:
    df = pd.read_parquet(ARCHIVO_DATASET)

    FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]

    X = df[FEATURES]
    y = df['target']

    print(f"✅ Dataset cargado: {len(df):,} filas x {len(FEATURES)} features")
    return X, y


# ============================================================
# WALK FORWARD VALIDATION
# ============================================================
def walk_forward(X: pd.DataFrame, y: pd.Series,
                 modelo, nombre: str,
                 escalar: bool = False) -> dict:
    """
    Divide los datos en N ventanas cronológicas.
    Entrena siempre con datos pasados, evalúa con datos futuros.
    Nunca mezcla pasado y futuro.
    """
    n          = len(X)
    tam_test   = n // (N_VENTANAS + 1)
    resultados = []

    print(f"\n{'─'*55}")
    print(f"  {nombre}")
    print(f"{'─'*55}")
    print(f"  {'Ventana':<10} {'Precisión':>10} {'Recall':>8} "
          f"{'F1':>8} {'AUC':>8} {'Ops':>6}")
    print(f"  {'─'*50}")

    for i in range(N_VENTANAS):
        # El entrenamiento crece con cada ventana
        fin_train  = tam_test * (i + 1)
        fin_test   = tam_test * (i + 2)

        X_train = X.iloc[:fin_train]
        y_train = y.iloc[:fin_train]
        X_test  = X.iloc[fin_train:fin_test]
        y_test  = y.iloc[fin_train:fin_test]

        # Escalado (solo para Regresión Logística)
        if escalar:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # Entrenamiento
        modelo.fit(X_train, y_train)

        # Predicción con umbral de probabilidad
        proba    = modelo.predict_proba(X_test)[:, 1]
        y_pred   = (proba >= UMBRAL_PROB).astype(int)

        # Métricas solo donde el modelo decidió operar
        mask     = y_pred == 1
        n_ops    = mask.sum()

        if n_ops > 0:
            precision = precision_score(y_test[mask], y_pred[mask],
                                        zero_division=0)
            recall    = recall_score(y_test, y_pred, zero_division=0)
            f1        = f1_score(y_test, y_pred, zero_division=0)
            auc       = roc_auc_score(y_test, proba)
        else:
            precision = recall = f1 = auc = 0.0

        resultados.append({
            'ventana':   i + 1,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'auc':       auc,
            'n_ops':     n_ops
        })

        fecha_inicio = X.index[fin_train].strftime('%m/%Y')
        fecha_fin    = X.index[fin_test - 1].strftime('%m/%Y')
        label        = f"V{i+1} {fecha_inicio}-{fecha_fin}"

        print(f"  {label:<18} {precision:>9.3f} {recall:>8.3f} "
              f"{f1:>8.3f} {auc:>8.3f} {n_ops:>6,}")

    # Promedios finales
    df_res    = pd.DataFrame(resultados)
    print(f"  {'─'*50}")
    print(f"  {'PROMEDIO':<18} "
          f"{df_res['precision'].mean():>9.3f} "
          f"{df_res['recall'].mean():>8.3f} "
          f"{df_res['f1'].mean():>8.3f} "
          f"{df_res['auc'].mean():>8.3f} "
          f"{df_res['n_ops'].mean():>6.0f}")

    return {
        'nombre':    nombre,
        'modelo':    modelo,
        'precision': df_res['precision'].mean(),
        'recall':    df_res['recall'].mean(),
        'f1':        df_res['f1'].mean(),
        'auc':       df_res['auc'].mean(),
        'resultados': df_res
    }


# ============================================================
# IMPORTANCIA DE FEATURES (Random Forest y XGBoost)
# ============================================================
def graficar_importancia(modelo, feature_names: list, nombre: str):
    importancias = pd.Series(
        modelo.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    importancias.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f"Importancia de Features — {nombre}")
    ax.set_xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig(f'importancia_{nombre.lower().replace(" ", "_")}.png',
                dpi=150)
    plt.show()
    print(f"✅ Gráfico guardado")


# ============================================================
# COMPARACIÓN FINAL DE MODELOS
# ============================================================
def comparar_modelos(resultados: list):
    print(f"\n{'='*60}")
    print("   COMPARACIÓN FINAL DE MODELOS")
    print(f"{'='*60}")
    print(f"  {'Modelo':<22} {'Precisión':>10} {'F1':>8} "
          f"{'AUC':>8} {'Ops/ventana':>12}")
    print(f"  {'─'*56}")

    mejor = max(resultados, key=lambda x: x['precision'])

    for r in resultados:
        marca = " ← MEJOR" if r['nombre'] == mejor['nombre'] else ""
        print(f"  {r['nombre']:<22} {r['precision']:>10.3f} "
              f"{r['f1']:>8.3f} {r['auc']:>8.3f} "
              f"{r['resultados']['n_ops'].mean():>12.0f}{marca}")

    print(f"{'='*60}")
    print(f"\n  Modelo seleccionado: {mejor['nombre']}")
    print(f"  Precisión promedio:  {mejor['precision']:.3f}")
    print(f"  Esto significa que cuando el modelo dice COMPRAR,")
    print(f"  acierta el {mejor['precision']*100:.1f}% de las veces")
    return mejor


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("🤖 Entrenamiento de modelos ML\n")

    # Cargamos dataset
    X, y = cargar_dataset()

    FEATURES = list(X.columns)

    # Definimos los tres modelos
    modelos = [
        (LogisticRegression(max_iter=1000, class_weight='balanced'),
         "Regresión Logística", True),

        (RandomForestClassifier(n_estimators=200, max_depth=6,
                                min_samples_leaf=50,
                                class_weight='balanced',
                                random_state=42, n_jobs=-1),
         "Random Forest", False),

        (XGBClassifier(n_estimators=300, max_depth=4,
                       learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=50,
                       scale_pos_weight=1, eval_metric='logloss',
                       random_state=42, n_jobs=-1),
         "XGBoost", False),
    ]

    resultados = []

    for modelo, nombre, escalar in modelos:
        r = walk_forward(X, y, modelo, nombre, escalar)
        resultados.append(r)

    # Comparación final
    mejor = comparar_modelos(resultados)

    # Importancia de features para RF y XGBoost
    print("\n📊 Generando gráficos de importancia de features...")
    for r in resultados:
        if r['nombre'] in ["Random Forest", "XGBoost"]:
            # Reentrenamos con todos los datos para el gráfico
            if r['nombre'] == "Random Forest":
                mod_final = RandomForestClassifier(
                    n_estimators=200, max_depth=6,
                    min_samples_leaf=50, class_weight='balanced',
                    random_state=42, n_jobs=-1)
            else:
                mod_final = XGBClassifier(
                    n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=50,
                    eval_metric='logloss', random_state=42, n_jobs=-1)

            mod_final.fit(X, y)
            graficar_importancia(mod_final, FEATURES, r['nombre'])

    print("\n✅ Entrenamiento completado")