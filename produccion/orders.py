import time
import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.error import ClientError
from config import BINANCE_KEY, BINANCE_SECRET, PAR
from notifier import enviar

config_logging(logging, logging.DEBUG)

# Variable global para almacenar la única instancia del cliente
_binance_client = None

def get_client():
    """
    Retorna la instancia del cliente de Binance.
    Aplica el patrón Singleton para evitar crear múltiples conexiones HTTP.
    """
    global _binance_client
    if _binance_client is None:
        # Solo se inicializa la primera vez que se llama
        _binance_client = Client(BINANCE_KEY, BINANCE_SECRET)
    return _binance_client

def nueva_orden(par: str, side: str, price: float, quantity: float) -> dict:
    """
    Envía una orden LIMIT a Binance.
    side: 'BUY' o 'SELL'
    Retorna el response de la orden o None si falla.
    """
    client = get_client()
    params = {
        "symbol": par,
        "side": side,
        "type": "LIMIT",
        "timeInForce": "GTC",
        "quantity": quantity,
        "price": price
    }
    try:
        response = client.new_order(**params)
        logging.info(response)
        return response
    except ClientError as error:
        mensaje = f"Error al enviar orden {side}: status {error.status_code} | código {error.error_code} | {error.error_message}"
        logging.error(mensaje)
        enviar(f"⚠️ {mensaje}")
        return None


def esperar_confirmacion(par: str, order_id: str, timeout: int = 240, intervalo: int = 1) -> dict | None:
    """
    Espera hasta que una orden alcance un estado final (FILLED, CANCELED, REJECTED).
    Si expira el timeout, cancela la orden activa y retorna el estado final.
    """
    client = get_client()
    tiempo = 0
    
    while tiempo < timeout:
        try:
            orden = client.get_order(symbol=par, orderId=str(order_id))
            estado = orden['status']
            
            # Estados finales: salimos del bucle
            if estado in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED' ]:
                return orden
            
            # Si es NEW o PARTIALLY_FILLED, seguimos esperando...
        
        except Exception as error: # Reemplazar Exception por la excepción específica de tu librería Binance
            logging.error(f"Error consultando orden {order_id}: {error}")
            
        time.sleep(intervalo)
        tiempo += intervalo

    # --- MANEJO DE TIMEOUT ---
    logging.warning(f"Orden {order_id} no completada en {timeout}s. Intentando cancelar...")
    
    try:
        client.cancel_order(symbol=par, orderId=str(order_id))
        logging.info(f"Orden {order_id} cancelada por timeout.")
    except Exception as e:
        # Puede fallar si la orden se llenó exactamente en este instante
        logging.error(f"No se pudo cancelar la orden {order_id} (quizás ya se completó): {e}")

    # --- ESTADO FINAL ---
    # Consultamos por última vez para saber exactamente qué porción se ejecutó
    try:
        orden_final = client.get_order(symbol=par, orderId=str(order_id))
        return orden_final
    except Exception as e:
        logging.error(f"Error al recuperar estado final de la orden {order_id}: {e}")
        return None


def ejecutar_compra(par: str, price: float, quantity: float) -> dict | None:
    orden = nueva_orden(par, 'BUY', price, quantity)
    if orden is None:
        return None

    orden_final = esperar_confirmacion(par, orden['orderId'], timeout=240)
    
    if not orden_final:
        enviar("⚠️ Error crítico: Se perdió el rastro de la orden de COMPRA.")
        return None

    estado = orden_final['status']
    ejecutado = float(orden_final.get('executedQty', 0))

    if estado == 'FILLED':
        enviar(f"✅ COMPRA completada\nPrecio: {price}\nCantidad: {ejecutado}")
    elif estado == 'CANCELED' and ejecutado > 0:
        enviar(f"⚠️ COMPRA parcial (Cancelada)\nSe compraron {ejecutado} de {quantity} al precio {price}")
    elif estado == 'CANCELED' and ejecutado == 0:
        enviar(f"❌ COMPRA cancelada sin ejecuciones por timeout.")
    else:
        enviar(f"⚠️ Estado inusual en COMPRA: {estado}. Revisar manualmente.")
        
    return orden_final


def ejecutar_venta(par: str, price: float, quantity: float, motivo: str = "VENTA") -> dict | None:
    orden = nueva_orden(par, 'SELL', price, quantity)
    if orden is None:
        return None

    orden_final = esperar_confirmacion(par, orden['orderId'], timeout=120)
    
    if not orden_final:
        enviar(f"⚠️ Error crítico: Se perdió el rastro de la orden de {motivo}.")
        return None

    estado = orden_final['status']
    ejecutado = float(orden_final.get('executedQty', 0))
    total_obtenido = round(price * ejecutado, 2)

    if estado == 'FILLED':
        enviar(f"🔴 {motivo} completada\nPrecio: {price}\nCantidad: {ejecutado}\nTotal: {total_obtenido} USDT")
    elif estado == 'CANCELED' and ejecutado > 0:
        enviar(f"⚠️ {motivo} parcial (Cancelada)\nSe vendieron {ejecutado} de {quantity}\nTotal: {total_obtenido} USDT")
    elif estado == 'CANCELED' and ejecutado == 0:
        enviar(f"❌ {motivo} cancelada sin ejecuciones por timeout.")
    else:
        enviar(f"⚠️ Estado inusual en {motivo}: {estado}. Revisar manualmente.")

    return orden_final
