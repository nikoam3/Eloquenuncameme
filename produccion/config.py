"""
Lee el .env, valida y expone las variables al resto del proyecto.
Falla de manera segura si falta configuración esencial.
"""
from dotenv import load_dotenv
import os
import sys

# Cargar variables de entorno
load_dotenv()

def get_env_variable(var_name: str) -> str:
    """
    Obtiene una variable de entorno y lanza un error descriptivo si no existe.
    """
    value = os.getenv(var_name)
    if not value or not value.strip():
        print(f"🛑 Error Crítico de Configuración: La variable de entorno '{var_name}' no está definida o está vacía.")
        sys.exit(1) # Detiene la ejecución del bot con código de error
    return value.strip()

# --- VALIDACIÓN DE VARIABLES ---

# 1. Credenciales de Binance
BINANCE_KEY = get_env_variable("BINANCE_KEY")
BINANCE_SECRET = get_env_variable("BINANCE_SECRET")

# 2. Credenciales de Telegram
TELEGRAM_TOKEN = get_env_variable("TELEGRAM_TOKEN")

# 3. Validación específica para el ID del Chat (debe ser numérico)
_chat_id_raw = get_env_variable("TELEGRAM_CHAT_ID")
try:
    TELEGRAM_CHAT_ID = int(_chat_id_raw)
except ValueError:
    print(f"🛑 Error Crítico de Configuración: 'TELEGRAM_CHAT_ID' debe ser un número entero válido. Valor recibido: '{_chat_id_raw}'")
    sys.exit(1)

# Configuración del par a operar
PAR = "ETHUSDT"
SYMBOL = "ETH"
INTERVALO = "15m"
DEBUG = True #True en desarrollo, False en producción

print("✅ Configuración cargada y validada correctamente.")
