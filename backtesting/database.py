import sqlite3
import requests
import pandas as pd
import pandas_ta as ta
import datetime as dt
from pathlib import Path

DB_PATH = "crypto_data.db"

PARES = [
    "BTCUSDT",
    "ETHUSDT", 
    "BNBUSDT",
    "SOLUSDT",
    "GALAUSDT"
]

INTERVALOS = ["5m", "15m", "1h"]


def get_connection():
    return sqlite3.connect(DB_PATH)


def crear_tablas():
    """Crea la estructura de la base de datos si no existe"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS velas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            par         TEXT    NOT NULL,
            intervalo   TEXT    NOT NULL,
            time        INTEGER NOT NULL,
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      REAL    NOT NULL,
            UNIQUE(par, intervalo, time)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_par_intervalo_time 
        ON velas(par, intervalo, time)
    """)

    conn.commit()
    conn.close()
    print("✅ Tablas creadas correctamente")


def descargar_velas(par: str, intervalo: str, dias: int) -> list:
    
    """Descarga velas desde Binance API"""
    url = 'https://api.binance.com/api/v3/klines'
    end_time   = int(dt.datetime.now().timestamp() * 1000)
    start_time = int((dt.datetime.now() - dt.timedelta(days=dias)).timestamp() * 1000)
    todas = []
    current = start_time

    while current < end_time:
        r = requests.get(url, params={
            'symbol':    par,
            'interval':  intervalo,
            'startTime': current,
            'endTime':   end_time,
            'limit':     1000
        })
        datos = r.json()
        if not datos:
            break
        todas.extend(datos)
        current = datos[-1][0] + 1
        print(f"  {par} {intervalo}: {len(todas)} velas descargadas...", end='\r')

    return todas


def guardar_velas(par: str, intervalo: str, velas: list):
    """Guarda velas en la base de datos, ignora duplicados"""
    conn = get_connection()
    cursor = conn.cursor()

    registros = [
        (par, intervalo, int(v[0]), float(v[1]), float(v[2]),
         float(v[3]), float(v[4]), float(v[5]))
        for v in velas
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO velas 
        (par, intervalo, time, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, registros)

    conn.commit()
    conn.close()


def cargar_velas(par: str, intervalo: str, 
                 desde: str = None, hasta: str = None) -> pd.DataFrame:
    """
    Carga velas desde la base de datos como DataFrame.
    desde/hasta: formato 'YYYY-MM-DD'
    """
    conn = get_connection()

    query = "SELECT * FROM velas WHERE par=? AND intervalo=?"
    params = [par, intervalo]

    if desde:
        ts = int(dt.datetime.strptime(desde, '%Y-%m-%d').timestamp() * 1000)
        query  += " AND time >= ?"
        params.append(ts)

    if hasta:
        ts = int(dt.datetime.strptime(hasta, '%Y-%m-%d').timestamp() * 1000)
        query  += " AND time <= ?"
        params.append(ts)

    query += " ORDER BY time ASC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return df

    df.index = pd.to_datetime(df['time'], unit='ms')
    df = df.drop(['id', 'par', 'intervalo', 'time'], axis=1)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df


def actualizar_datos():
    """
    Descarga datos nuevos desde la última vela guardada.
    Esto es lo que vas a correr periódicamente.
    """
    conn = get_connection()
    cursor = conn.cursor()

    for par in PARES:
        for intervalo in INTERVALOS:
            # Buscamos la última vela guardada
            cursor.execute("""
                SELECT MAX(time) FROM velas 
                WHERE par=? AND intervalo=?
            """, (par, intervalo))

            ultimo = cursor.fetchone()[0]

            if ultimo is None:
                # Primera vez: descargamos 2 años
                print(f"\nDescarga inicial {par} {intervalo} (2 años)...")
                dias = 730
            else:
                # Ya tenemos datos: descargamos solo lo nuevo
                ultimo_dt = dt.datetime.fromtimestamp(ultimo / 1000)
                dias_faltantes = (dt.datetime.now() - ultimo_dt).days + 1
                print(f"\nActualizando {par} {intervalo} ({dias_faltantes} días nuevos)...")
                dias = dias_faltantes

            velas = descargar_velas(par, intervalo, dias)
            guardar_velas(par, intervalo, velas)
            print(f"  ✅ {par} {intervalo}: {len(velas)} velas guardadas")

    conn.close()


def info_base_datos():
    """Muestra un resumen de lo que hay en la base de datos"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT par, intervalo, COUNT(*) as total,
               MIN(time) as desde, MAX(time) as hasta
        FROM velas
        GROUP BY par, intervalo
        ORDER BY par, intervalo
    """)

    rows = cursor.fetchall()
    conn.close()

    print("\n" + "="*70)
    print(f"  {'Par':<12} {'Intervalo':<10} {'Velas':>8} {'Desde':<15} {'Hasta'}")
    print("="*70)
    for row in rows:
        par, intervalo, total, desde, hasta = row
        desde_dt = dt.datetime.fromtimestamp(desde / 1000).strftime('%Y-%m-%d')
        hasta_dt = dt.datetime.fromtimestamp(hasta / 1000).strftime('%Y-%m-%d')
        print(f"  {par:<12} {intervalo:<10} {total:>8,} {desde_dt:<15} {hasta_dt}")
    print("="*70)


if __name__ == "__main__":
    print("🗄️  Iniciando base de datos...\n")
    crear_tablas()
    actualizar_datos()
    info_base_datos()