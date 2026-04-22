import logging

SYMBOL = "XAUUSD"
MAGIC_NUMBER = 123456

# === CAPITAL SIMULADO ===
SIMULATED_BALANCE = 100.0   # Para cuentas de bajo monto (100 USD o menos)
USE_REAL_BALANCE = False
RESET_BALANCE_DAILY = False  # Mantiene balance día a día (es su cuenta real)

# === LOTE FIJO ===
FIXED_LOT_SIZE = 0.01  # 0.01 lote es óptimo para $100

# === SPREAD ===
SPREAD_POINTS = 30
SPREAD_COST_USD = SPREAD_POINTS * 0.01

# === SL/TP Y TRAILING ===
MIN_SL_USD = 0.5  # Reducido para mayor dinamismo
MAX_SL_USD = 3.5  # Aumentado para operaciones volatiles
ATR_MULTIPLIER_SL = 1.0  # Aumentado de 0.5 (ATR completo, no mitad)
MIN_RISK_REWARD_RATIO = 1.5        # Aumentado de 1.2 - mejor ratio ganancia
MAX_RISK_REWARD_RATIO = 2.5        # Aumentado de 2.0

# Trailing stop menos agresivo (permite que trades crezcan)
TRAILING_TRIGGER_PROFIT = 8.0      # Aumentado de 2.0 (activa a 8 USD ganancia)
TRAILING_DISTANCE_USD = 2.0        # Aumentado de 1.0 (menos cierre por ruido)

# === FILTROS MAS ESTRICTOS ===
MIN_CONFIDENCE = 0.65              # Mantenido: 0.55 → 0.65 (solo señales confiables)
MAX_SPREAD_USD = 1.5               # Restaurado: permitte operaciones (1.5 USD es aceptable)
MIN_ATR_USD = 1.2                  # Intermediario: 1.0 → 1.2 (cierta volatilidad)
MIN_MOVE_TO_TRADE_USD = 2.0        # Reducido de 4.5: TP mínimo realista para $100
MAX_DAILY_LOSSES = 8               # Mantenido: permite operaciones hasta parada
MAX_DAILY_TRADES = 15              # Aumentado de 5 → 15 (necesita volumen para $100)

# === CONTROL DE PÉRDIDAS CONSECUTIVAS ===
MAX_CONSECUTIVE_LOSSES = 3         # Reducido de 4 - reacciona mas rapido
CONSECUTIVE_LOSS_PAUSE_HOURS = 1   # Reducido de 2 - pausa mas corta

# === DRAWDOWN ===
MAX_DRAWDOWN_PERCENT = 40.0        # Para $100: pérdida máxima $40 USD
REDUCE_RISK_AT_DRAWDOWN = 25.0     # Reduce lote si drawdown > 25%

# === HORARIO AMPLIADO ===
USE_HOUR_FILTER = True
ACTIVE_HOURS_START = 5
ACTIVE_HOURS_END = 20              # Hasta las 8 PM Colombia

# === TIMEFRAMES ===
TIMEFRAME_MAIN = "H1"
TIMEFRAME_TREND = "H4"

# === ARCHIVOS ===
MODEL_PATH = "rf_model.pkl"
FEATURES_PATH = "feature_columns.pkl"
TRADES_CSV = "trades_log.csv"
BALANCE_FILE = "simulated_balance.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger("XAUUSD_BOT")