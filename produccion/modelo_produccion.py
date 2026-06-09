import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from features import construir_features

# ============================================================
# CONFIGURACIÓN
# ============================================================
UMBRAL_PROB   = 0.60
MODELO_PATH   = 'modelo_lr.joblib'
SCALER_PATH   = 'scaler_lr.joblib'
PAR           = "ETHUSDT"


# ============================================================
# ENTRENAMIENTO Y GUARDADO DEL MODELO
# ============================================================
def entrenar_y_guardar():
    """
    Entrena el modelo con todos los datos disponibles
    y lo guarda en disco para usarlo en producción.
    """
    print("Cargando dataset...")
    df = pd.read_parquet('dataset_ml.parquet')

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

    print(f"Entrenando con {len(df):,} muestras...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )
    modelo.fit(X_scaled, y)

    # Guardamos modelo y scaler
    joblib.dump(modelo, MODELO_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"✅ Modelo guardado en {MODELO_PATH}")
    print(f"✅ Scaler guardado en {SCALER_PATH}")
    return modelo, scaler


# ============================================================
# CARGA DEL MODELO
# ============================================================
def cargar_modelo():
    """
    Carga el modelo y scaler desde disco.
    Si no existen, los entrena primero.
    """
    if not os.path.exists(MODELO_PATH) or \
       not os.path.exists(SCALER_PATH):
        print("Modelo no encontrado, entrenando...")
        return entrenar_y_guardar()

    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Modelo cargado desde disco")
    return modelo, scaler


# ============================================================
# PREDICCIÓN EN TIEMPO REAL
# ============================================================
def predecir(modelo, scaler, par: str = PAR) -> dict:
    """
    Calcula las features del momento actual y
    devuelve la probabilidad de que la próxima
    operación sea ganadora.

    Retorna un diccionario con:
        - prob:    probabilidad de operación ganadora (0-1)
        - operar:  True si supera el umbral
        - señales: valores actuales de las features clave
    """
    FEATURES = [
        'close_vs_ema200', 'ema50_vs_ema200',
        'atr_relativo', 'atr_tendencia', 'bb_ancho',
        'rsi',
        'adx',
        'hora', 'dia_semana',
        'tendencia_1h', 'rsi_1h', 'adx_1h', 'roc_1h'
    ]

    # Construimos features con datos actuales
    features_df, _ = construir_features(par)

    # Tomamos la última fila: el momento actual
    ultima = features_df[FEATURES].iloc[-1:]

    # Escalamos y predecimos
    X_scaled = scaler.transform(ultima)
    prob = modelo.predict_proba(X_scaled)[0][1]

    # Valores legibles para el log
    señales = {
        'rsi_15m':       round(float(ultima['rsi'].iloc[0]), 1),
        'rsi_1h':        round(float(ultima['rsi_1h'].iloc[0]), 1),
        'adx_1h':        round(float(ultima['adx_1h'].iloc[0]), 1),
        'tendencia_1h':  int(ultima['tendencia_1h'].iloc[0]),
        'atr_relativo':  round(float(ultima['atr_relativo'].iloc[0]), 4),
        'close_vs_ema200': round(float(ultima['close_vs_ema200'].iloc[0]), 4),
    }

    return {
        'prob':    round(prob, 4),
        'operar':  prob >= UMBRAL_PROB,
        'señales': señales
    }