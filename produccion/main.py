# Agregar junto a los otros imports
from modelo_produccion import cargar_modelo
import time
import datetime as dt
from market import get_decimales
from features import construir_features
from strategy import (
    get_estado,
    evaluar_compra,
    evaluar_venta_ml,
    evaluar_stop_loss,
    actualizar_trailing_stop,
    incrementar_ciclo
)
from notifier import enviar
from config import PAR, DEBUG


def imprimir_estado(ciclos: int, data):
    """Imprime el estado actual del bot en consola"""
    estado = get_estado()
    print(
        f"\n{'='*40}"
        f"\n Ciclo:      {ciclos}"
        f"\n Compras:    {estado['compras']}"
        f"\n Ventas:     {estado['ventas']}"
        f"\n Stop Loss:  {estado['stops_loss']}"
        f"\n Take Profit:{estado['take_profit']}"
        f"\n RSI actual: {round(float(data['RSI'].iloc[-1]), 2)}"
        f"\n Precio:     {float(data['Close'].iloc[-1])}"
        f"\n Stop Price: {estado['stop_price']}"
        f"\n TP Price:   {estado['stop_take_price']}"
        f"\n{'='*40}"
    )
def dormir_hasta_proxima_ejecucion(minutos_intervalo: int = 15):
    """
    Calcula el tiempo exacto hasta la próxima ventana de ejecución 
    y pausa el bot, liberando recursos del procesador.
    """
    now = dt.datetime.now()
    
    # Calculamos cuántos minutos faltan para el próximo múltiplo del intervalo
    # Ej: Si son y 12, y el intervalo es 15, faltan 3 minutos.
    minutos_faltantes = minutos_intervalo - (now.minute % minutos_intervalo)
    
    # Construimos el objeto datetime exacto de la próxima ejecución
    proximo_target = now + dt.timedelta(minutes=minutos_faltantes)
    
    # Forzamos los segundos a 1 (como tenías en tu lógica) y limpiamos microsegundos
    proximo_target = proximo_target.replace(second=1, microsecond=0)
    
    # Recalculamos el 'now' justo antes de dormir por mayor precisión
    segundos_a_dormir = (proximo_target - dt.datetime.now()).total_seconds()
    
    if segundos_a_dormir > 0:
        print(f"⏳ Esperando próxima vela. Durmiendo {int(segundos_a_dormir)}s hasta las {proximo_target.strftime('%H:%M:%S')}")
        time.sleep(segundos_a_dormir)

def main():
    enviar("🤖 Bot iniciado")
    
    try:
        decimal_price, decimal_quantity = get_decimales(PAR)
        ciclos = 0

        # Cargamos el modelo ML (nuevo)
        print("Cargando modelo ML...")
        modelo_ml, scaler_ml = cargar_modelo()
        print("✅ Modelo ML listo")
    except Exception as e:
        mensaje = f"❌ Error al cargar modelo o decimales: {str(e)}"
        print(mensaje)
        enviar(mensaje)
        return

    while True:
        # Cuando time.sleep() termina, sabemos que es el momento exacto
        dormir_hasta_proxima_ejecucion(minutos_intervalo=15)
        estado = get_estado()
        try:
            # 1. Obtenemos datos con indicadores
            features, data = construir_features()

            # 2. Evaluamos señales en orden de prioridad
            evaluar_stop_loss(data, decimal_price, decimal_quantity)

            if not estado["buy"]:   # solo si no hay posición abierta
                evaluar_compra(decimal_price, decimal_quantity, modelo_ml, scaler_ml, features)
            else:
                actualizar_trailing_stop(data, decimal_price)
                evaluar_venta_ml(decimal_price, decimal_quantity, modelo_ml, scaler_ml, features)
            
            # 3. Actualizamos contadores
            incrementar_ciclo()
            ciclos += 1

            if DEBUG:
                # 4. Mostramos estado y gráfico
                imprimir_estado(ciclos, data)

        except Exception as e:
            mensaje = f"❌ Error en ciclo {ciclos}: {str(e)}"
            print(mensaje)
            enviar(mensaje)

        # Esperamos 2 segundos para no ejecutar dos veces en el mismo minuto
        time.sleep(2)

    


if __name__ == "__main__":
    main()