import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print("ERROR: No se pudo inicializar MT5")
    exit()

print("MT5 inicializado correctamente")

# Activar símbolo
if not mt5.symbol_select("XAUUSD", True):
    print("ERROR: No se pudo activar XAUUSD")
else:
    print("XAUUSD activado")

# Ver información del símbolo
info = mt5.symbol_info("XAUUSD")
if info:
    print(f"Nombre: {info.name},  punto: {info.point},  spread: {info.spread}")
else:
    print("No se obtuvo info del símbolo")

# Probar copy_rates_range (últimos 1000 datos)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M5, pd.Timestamp.now() - pd.Timedelta(days=30), pd.Timestamp.now())
if rates is None:
    print("ERROR: copy_rates_range devolvió None")
    print(f"Último error: {mt5.last_error()}")
else:
    print(f"Descargadas {len(rates)} velas en M5 de los últimos 30 días")
    if len(rates) > 0:
        df = pd.DataFrame(rates)
        print(df.head())
    else:
        print("No hay datos (0 velas)")

mt5.shutdown()