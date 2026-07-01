"""
esto lo hizo copilot para entender si el modelo de produccion 
es similar al del backtesting con 4 ventanas
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


FEATURES = [
    'close_vs_ema200', 'ema50_vs_ema200', 'ema9_vs_ema26',
    'rsi',
    'atr_relativo', 'atr_tendencia', 'bb_ancho',
    'adx', 'rsi_pendiente_vs_precio',
    'macd_histograma',
    'hora', 'dia_semana',
    'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
]


def cargar_dataset(path='dataset_ml.parquet'):
    df = pd.read_parquet(path)
    X = df.loc[:, FEATURES].copy()
    y = df['target'].astype(int)
    print(f"Dataset cargado: {len(df):,} filas, {len(FEATURES)} features")
    return df, X, y


def entrenar_modelo(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo.fit(X_scaled, y)

    pred = modelo.predict(X_scaled)
    proba = modelo.predict_proba(X_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'auc': roc_auc_score(y, proba),
    }

    return modelo, scaler, metrics


def walk_forward(X, y, n_ventanas=4):
    n = len(X)
    tam_test = n // (n_ventanas + 1)
    resultados = []

    for i in range(n_ventanas):
        fin_train = tam_test * (i + 1)
        fin_test = tam_test * (i + 2)

        X_train = X.iloc[:fin_train]
        y_train = y.iloc[:fin_train]
        X_test = X.iloc[fin_train:fin_test]
        y_test = y.iloc[fin_train:fin_test]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
        modelo.fit(X_train_scaled, y_train)

        pred = modelo.predict(X_test_scaled)
        proba = modelo.predict_proba(X_test_scaled)[:, 1]

        resultados.append({
            'ventana': i + 1,
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'auc': roc_auc_score(y_test, proba),
            'n_test': len(y_test),
        })

    df_res = pd.DataFrame(resultados)
    resumen = {
        'accuracy_mean': df_res['accuracy'].mean(),
        'precision_mean': df_res['precision'].mean(),
        'recall_mean': df_res['recall'].mean(),
        'f1_mean': df_res['f1'].mean(),
        'auc_mean': df_res['auc'].mean(),
        'n_test_mean': df_res['n_test'].mean(),
    }

    return df_res, resumen


if __name__ == '__main__':
    print('=' * 70)
    print('COMPARACIÓN DE FLUJOS: PRODUCCIÓN vs WALK-FORWARD')
    print('=' * 70)

    df, X, y = cargar_dataset()

    print('\n1) Flujo de producción (entrenar con todo el dataset)')
    modelo, scaler, metrics_prod = entrenar_modelo(X, y)
    print('   Métricas sobre el mismo dataset usado para entrenar:')
    for nombre, valor in metrics_prod.items():
        print(f'   - {nombre:10}: {valor:.4f}')

    print('\n2) Flujo de backtesting (4 ventanas cronológicas)')
    df_res, resumen_wf = walk_forward(X, y, n_ventanas=4)
    print('   Promedio de las 4 ventanas:')
    for nombre, valor in resumen_wf.items():
        print(f'   - {nombre:12}: {valor:.4f}')

    print('\n3) Resumen comparativo')
    print('   (Nota: el modelo de producción se evalúa sobre el dataset completo,')
    print('    mientras que el walk-forward mide performance out-of-sample por ventana).')
    print('   - Producción (in-sample):')
    print(f'       accuracy={metrics_prod["accuracy"]:.4f}   '
          f'f1={metrics_prod["f1"]:.4f}   auc={metrics_prod["auc"]:.4f}')
    print('   - Walk-forward (promedio 4 ventanas):')
    print(f'       accuracy={resumen_wf["accuracy_mean"]:.4f}   '
          f'f1={resumen_wf["f1_mean"]:.4f}   auc={resumen_wf["auc_mean"]:.4f}')

    print('\n4) Detalle por ventana')
    print(df_res.to_string(index=False, float_format='%.4f'))
    print('\n' + '=' * 70)
