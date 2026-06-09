from market import get_saldo, get_order_book
from orders import ejecutar_compra, ejecutar_venta
from notifier import enviar
from config import PAR, SYMBOL
from modelo_produccion import predecir
import json
import os
import logging 

ARCHIVO_ESTADO = "estado.json"

# Estado interno de la estrategia
estado_inicial = {
    "buy": False,
    "compras": 0,
    "ventas": 0,
    "stops_loss": 0,
    "take_profit": 0,
    "saldo": 0.0,
    "price_compra": 0.0,
    "price_take_profit": 0.0,
    "stop_take_price": 0.0,
    "stop_price": 0.0,
    "rsi_min": 100.0,
    "rsi_max": 0.0,
    "cicles_cont": 0
}

def cargar_estado() -> dict:
    """
    Carga el estado desde el disco al arrancar el bot.
    Si el archivo no existe o está corrupto, retorna el estado por defecto.
    """
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, 'r') as f:
                estado_cargado = json.load(f)
            logging.info("✅ Estado recuperado exitosamente tras el reinicio.")
            
            # Combinamos el estado inicial con el cargado. 
            # Esto es vital por si en el futuro agregás nuevas claves al ESTADO_INICIAL.
            return {**estado_inicial, **estado_cargado}
            
        except json.JSONDecodeError:
            logging.error("🛑 Archivo de estado corrupto. Iniciando con estado de fábrica.")
            return estado_inicial.copy()
    else:
        logging.info("ℹ️ No se encontró estado previo. Iniciando de cero.")
        return estado_inicial.copy()

def guardar_estado(estado_actual: dict):
    """
    Guarda el estado en el disco.
    Utiliza una escritura atómica para evitar que el archivo se corrompa 
    si el bot crashea exactamente en el milisegundo en que está guardando.
    """
    archivo_temporal = ARCHIVO_ESTADO + ".tmp"
    try:
        # Escribimos primero en un archivo temporal
        with open(archivo_temporal, 'w') as f:
            json.dump(estado_actual, f, indent=4)
            
        # Reemplazamos el archivo original por el temporal (Operación atómica en SO)
        os.replace(archivo_temporal, ARCHIVO_ESTADO)
    except Exception as e:
        logging.error(f"Error crítico al guardar el estado: {e}")

# --- Uso en tu archivo strategy.py ---
# Al arrancar el bot, inicializas la variable así:
estado = cargar_estado()
guardar_estado(estado) # Guardamos inmediatamente para crear el archivo si no existe

def resetear_rsi():
    estado["rsi_min"] = 100.0
    estado["rsi_max"] = 0.0
    estado["cicles_cont"] = 0


def resetear_posicion():
    estado["stop_price"] = 0.0
    estado["stop_take_price"] = 0.0
    estado["price_take_profit"] = 0.0
    estado["price_compra"] = 0.0
    estado["buy"] = False
    resetear_rsi()


def evaluar_compra(decimal_price: int, 
                   decimal_quantity: int,
                   modelo_ml=None, scaler_ml=None):
    """
    Evalúa si corresponde comprar según los indicadores
    y confirma con el modelo ML si está disponible.
    """
    if estado["buy"]:
        return  # ya tenemos posición abierta, no compramos otra
    
    # Filtro ML: consultamos el modelo antes de comprar
    if modelo_ml is None or scaler_ml is None:
        return  # sin modelo no operamos

    prediccion = predecir(modelo_ml, scaler_ml, PAR)
    prob = prediccion['prob']

    print(f"  🧠 ML: prob={prob} | "
          f"RSI_1h={prediccion['señales']['rsi_1h']}")

    if not prediccion['operar']:
        return # el modelo no ve un contexto favorable, no compramos

    # Resto igual: consultar saldo, calcular cantidad, ejecutar
    estado["saldo"] = get_saldo("USDT")
    if estado["saldo"] <= 11:
        return

    price = float(
        get_order_book(PAR, decimal_price)['ask_price'].iloc[0]
    )

    if (estado["saldo"] * 0.25) <= 11:
        quantity = round(
            (estado["saldo"] * 0.999 / price), decimal_quantity
        )
    else:
        quantity = round(
            (estado["saldo"] * 0.25) / price, decimal_quantity
        )

    orden = ejecutar_compra(PAR, price, quantity)

    if orden:
        estado["price_compra"]      = price
        estado["price_take_profit"] = round(price * 1.01, decimal_price)
        estado["stop_price"]        = round(price * 0.985, decimal_price)
        estado["buy"]               = True
        estado["compras"]          += 1
        resetear_rsi()
        enviar(
            f"🟢 *COMPRA ejecutada*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Precio: `{price}`\n"
            f"📦 Cantidad: `{quantity}`\n"
            f"🛑 Stop Loss: `{estado['stop_price']}`\n"
            f"🎯 Take Profit: `{estado['price_take_profit']}`\n"
            f"🧠 Prob ML: `{prob*100:.1f}%`"
        )
    guardar_estado(estado)


def evaluar_venta_ml(modelo_ml=None, scaler_ml=None,
                     decimal_price: int = 6,
                     decimal_quantity: int = 6):
    """
    Cierra la posición si el modelo detecta que
    el contexto de mercado se revirtió.
    Solo actúa si hay posición abierta.
    """
    if not estado["buy"]:
        return

    if modelo_ml is None or scaler_ml is None:
        return

    prediccion = predecir(modelo_ml, scaler_ml, PAR)
    prob = prediccion['prob']

    UMBRAL_VENTA = 1 - 0.63  # = 0.37

    if prob >= UMBRAL_VENTA:
        return  # contexto sigue favorable, no vendemos

    # Contexto se revirtió → cerramos posición
    quantity_pos = get_saldo(SYMBOL)
    price = float(
        get_order_book(PAR, decimal_price)['bid_price'].iloc[0]
    )

    if (quantity_pos * price) < 10:
        return

    quantity = round(quantity_pos * 0.999, decimal_quantity)
    orden = ejecutar_venta(PAR, price, quantity,
                           motivo="VENTA ML")

    if orden:
        enviar(
            f"🔵 VENTA por señal ML\n"
            f"Precio: {price}\n"
            f"Prob actual: {prob*100:.1f}%\n"
            f"Contexto revertido"
        )
        estado["ventas"] += 1
        resetear_posicion()
    guardar_estado(estado)



def evaluar_stop_loss(data, decimal_price: int, decimal_quantity: int):
    """
    Evalúa si el precio cayó por debajo del stop loss o del trailing stop.
    """
    close_actual = float(data['Close'].iloc[-2])

    # Si no hay posición abierta o no hay stops definidos, no hace nada
    if not estado["buy"]:
        return
    if estado["stop_price"] == 0 and estado["stop_take_price"] == 0:
        return

    # Verificamos si algún stop fue tocado
    stop_tocado = (
        (estado["stop_price"] > 0 and close_actual < estado["stop_price"]) or
        (estado["stop_take_price"] > 0 and close_actual < estado["stop_take_price"])
    )

    if not stop_tocado:
        return

    quantity_pos = get_saldo(SYMBOL)
    price = float(get_order_book(PAR, decimal_price)['bid_price'].iloc[0])

    if (quantity_pos * price) < 10:
        return

    quantity = round(quantity_pos * 0.999, decimal_quantity)

    # Determinamos el motivo para el mensaje
    if close_actual < estado["stop_price"]:
        motivo = "STOP LOSS"
        estado["stops_loss"] += 1
    else:
        motivo = "TAKE PROFIT"
        estado["take_profit"] += 1

    orden = ejecutar_venta(PAR, price, quantity, motivo=motivo)

    if orden:
        resetear_posicion()
    guardar_estado(estado)



def actualizar_trailing_stop(data, decimal_price: int):
    """
    Actualiza el trailing stop a medida que el precio sube.
    Tres niveles de protección según cuánto subió desde la compra.
    """
    if not estado["buy"]:
        return

    close_actual = float(data['Close'].iloc[-2])
    price_compra = estado["price_compra"]

    if close_actual <= estado["price_take_profit"]:
        return  # El precio no superó el último máximo registrado

    # Actualizamos el máximo
    estado["price_take_profit"] = close_actual

    # Nivel 1: ganancia menor al 1% → stop ajustado al 0.1% abajo
    if close_actual < (price_compra * 1.01):
        estado["stop_take_price"] = round(close_actual * 0.999, decimal_price)

    # Nivel 2: ganancia entre 1% y 2% → stop más holgado
    elif close_actual < (price_compra * 1.02):
        estado["stop_take_price"] = round(close_actual * 0.995, decimal_price)

    # Nivel 3: ganancia mayor al 2% → stop más holgado todavía
    else:
        estado["stop_take_price"] = round(close_actual * 0.99, decimal_price)
    guardar_estado(estado)



def incrementar_ciclo():
    """
    Cada 50 ciclos sin operación resetea los mínimos/máximos de RSI
    para evitar que queden valores viejos bloqueando nuevas entradas.
    """
    estado["cicles_cont"] += 1
    if estado["cicles_cont"] >= 50:
        resetear_rsi()
    guardar_estado(estado)
