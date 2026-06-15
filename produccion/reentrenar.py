"""
Ejecutar este script una vez por mes para mantener
el modelo actualizado con datos recientes.
"""
from database import actualizar_datos
from features import construir_features, crear_target
from modelo_produccion import entrenar_y_guardar

STOP_LOSS   = 0.025
TAKE_PROFIT = 0.010
MAX_VELAS   = 96

def reentrenar():
    print("="*50)
    print("  REENTRENAMIENTO MENSUAL DEL MODELO")
    print("="*50)

    # 1. Actualizamos la base de datos con velas nuevas
    print("\n1. Actualizando base de datos...")
    actualizar_datos()

    # 2. Reconstruimos features con datos actualizados
    print("\n2. Construyendo features actualizadas...")
    features_df, df_15m = construir_features("ETHUSDT", True)

    # 3. Recalculamos el target
    print("\n3. Calculando target...")
    target = crear_target(
        df_15m.loc[features_df.index],
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        max_velas=MAX_VELAS
    )
    df_ml = features_df.join(target).dropna()

    # 4. Guardamos dataset actualizado
    df_ml.to_parquet('dataset_ml.parquet')
    print(f"   Dataset actualizado: {len(df_ml):,} filas")

    # 5. Reentrenamos y guardamos el modelo
    print("\n4. Reentrenando modelo...")
    entrenar_y_guardar()

    print("\n✅ Reentrenamiento completado")
    print("   El bot usará el nuevo modelo en el próximo ciclo")

if __name__ == "__main__":
    reentrenar()