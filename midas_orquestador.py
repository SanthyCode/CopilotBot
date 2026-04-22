import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import json
import os
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SCALPING_BOT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración
SYMBOL = "XAUUSD"
INITIAL_BALANCE = 100.0
TIMEFRAME = mt5.TIMEFRAME_M15
LOT_SIZE = 0.01
MAX_DAILY_TRADES = 20

# SL/TP para scalping
SL_USD = 1.5
TP_USD = 3.0

MAX_SPREAD_USD = 1.0
MIN_ATR_USD = 0.8

simulated_balance = INITIAL_BALANCE


class ScalpingBot:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError("No se pudo inicializar MT5")
        self.trades_today = 0
        self.last_reset = datetime.now().date()
        self.state_file = "bot_state.json"
        self.open_ticket = None  # SOLO una orden a la vez
        self._load_state()
        logger.info(f"Bot inicializado | Capital: {self._get_balance():.2f} USD")

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.trades_today = data.get('trades', 0)
        except:
            pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                'date': str(datetime.now().date()),
                'trades': self.trades_today
            }, f)

    def _reset_daily(self):
        global simulated_balance
        today = datetime.now().date()
        if today != self.last_reset:
            self.trades_today = 0
            self.last_reset = today
            self._save_state()
            logger.info(f"=== NUEVO DÍA === Balance: {simulated_balance:.2f} USD")

    def _get_balance(self):
        global simulated_balance
        return simulated_balance

    def _update_balance(self, new_balance):
        global simulated_balance
        simulated_balance = new_balance
        with open('simulated_balance.json', 'w') as f:
            json.dump({
                'balance': round(simulated_balance, 2),
                'date': str(datetime.now().date())
            }, f)

    def _has_open_position(self):
        """Verifica si hay orden abierta actualmente"""
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for pos in positions:
                return pos.ticket
        return None

    def _check_closed_position(self, ticket):
        """Verifica si una posición se cerró y retorna P&L"""
        pos = mt5.positions_get(ticket=ticket)
        if pos:
            return None  # Aún abierta

        # Cerrada - obtener P&L
        deals = mt5.history_deals_get(ticket=ticket)
        if deals:
            d = deals[-1]
            pnl = d.profit
            return pnl
        return None

    def _get_ema(self, series, period):
        if len(series) < period:
            return None
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]

    def _get_rsi(self, series, period=14):
        if len(series) < period:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _get_atr(self, rates, period=14):
        if rates is None or len(rates) < period:
            return None
        df = pd.DataFrame(rates)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        atr = df['tr'].rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else None

    def _get_bollinger_bands(self, series, period=20, std_dev=2):
        if len(series) < period:
            return None, None, None
        bb_middle = series.rolling(period).mean()
        bb_std = series.rolling(period).std()
        bb_upper = bb_middle + (std_dev * bb_std)
        bb_lower = bb_middle - (std_dev * bb_std)
        return bb_upper.iloc[-1], bb_middle.iloc[-1], bb_lower.iloc[-1]

    def _check_signal(self):
        """Verifica señal de scalping"""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
        if rates is None or len(rates) < 20:
            return None, None

        df = pd.DataFrame(rates)
        close = df['close']
        high = df['high']
        low = df['low']

        ema5 = self._get_ema(close, 5)
        ema10 = self._get_ema(close, 10)
        rsi = self._get_rsi(close, 14)
        bb_upper, bb_middle, bb_lower = self._get_bollinger_bands(close, 20, 2)

        if ema5 is None or ema10 is None or rsi is None:
            return None, None

        last_close = close.iloc[-1]
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]

        # COMPRA
        if ema5 > ema10:
            if 40 < rsi < 70:
                if last_low < bb_lower and last_close > bb_lower:
                    return 'buy', 0.75
                elif last_close < bb_middle and rsi < 60:
                    return 'buy', 0.65

        # VENTA
        elif ema5 < ema10:
            if 30 < rsi < 60:
                if last_high > bb_upper and last_close < bb_upper:
                    return 'sell', 0.75
                elif last_close > bb_middle and rsi > 40:
                    return 'sell', 0.65

        return None, None

    def _place_order(self, direction):
        """Coloca una orden"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if not symbol_info:
            logger.error("Símbolo no encontrado")
            return None

        point = symbol_info.point
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return None

        sl_points = int(round(SL_USD / 0.01))
        tp_points = int(round(TP_USD / 0.01))

        if direction == 'buy':
            price = tick.ask
            sl_price = price - sl_points * point
            tp_price = price + tp_points * point
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl_price = price + sl_points * point
            tp_price = price - tp_points * point
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 30,
            "magic": 99999,
            "comment": f"SCALP-{direction.upper()}"
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Error orden: {mt5.last_error()}")
            return None

        logger.info(f"🔷 {direction.upper()} {LOT_SIZE} @ {price:.2f} | SL:{SL_USD:.1f} TP:{TP_USD:.1f}")
        return result.order

    def run(self):
        """Loop principal - SOLO 1 ORDEN A LA VEZ, LECTURA CONSTANTE"""
        logger.info("🚀 SCALPING BOT M15 - SINGLE ORDER MODE")
        logger.info(f"Config: SL={SL_USD} | TP={TP_USD} | 1 orden a la vez")

        account = mt5.account_info()
        if account:
            logger.info(f"Cuenta: {account.name} | Saldo Real: {account.balance:.2f}")

        while True:
            try:
                self._reset_daily()

                # Línea principal: siempre verifica si hay orden abierta
                if self.open_ticket is not None:
                    # Orden abierta: monitorear si se cerró
                    pnl = self._check_closed_position(self.open_ticket)
                    
                    if pnl is not None:
                        # Se cerró la orden
                        new_balance = self._get_balance() + pnl
                        self._update_balance(new_balance)

                        if pnl > 0:
                            logger.info(f"✓ +{pnl:.2f} USD | Balance: {new_balance:.2f} USD")
                        else:
                            logger.info(f"✗ {pnl:.2f} USD | Balance: {new_balance:.2f} USD")

                        self.open_ticket = None  # Orden cerrada, buscar nueva
                        self.trades_today += 1
                        self._save_state()

                    # Mientras hay orden abierta, solo monitorear (no buscar nuevas)
                    time.sleep(5)
                    continue

                # NO hay orden abierta: buscar nueva señal
                if self.trades_today >= MAX_DAILY_TRADES:
                    logger.info(f"Límite {MAX_DAILY_TRADES} ops alcanzado - esperando próximo día")
                    time.sleep(3600)
                    continue

                # Verificar spread
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    time.sleep(10)
                    continue

                spread_usd = (tick.ask - tick.bid) * 1.0
                if spread_usd > MAX_SPREAD_USD:
                    logger.debug(f"Spread alto {spread_usd:.2f}")
                    time.sleep(30)
                    continue

                # Verificar ATR
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
                if rates is None:
                    time.sleep(10)
                    continue

                atr = self._get_atr(rates, 14)
                if atr is None or atr < MIN_ATR_USD:
                    time.sleep(30)
                    continue

                # Buscar señal
                direction, confidence = self._check_signal()
                if direction is None:
                    logger.debug("Sin señal")
                    time.sleep(30)
                    continue

                logger.info(f"📊 Señal: {direction.upper()} (conf: {confidence:.2f})")

                # Colocar orden
                ticket = self._place_order(direction)
                if ticket:
                    self.open_ticket = ticket
                    logger.info(f"✅ Orden abierta (ticket: {ticket}) | Monitorando...")

                time.sleep(10)

            except Exception as e:
                logger.exception(f"Error: {e}")
                time.sleep(30)


if __name__ == "__main__":
    try:
        bot = ScalpingBot()
        bot.run()
    except Exception as e:
        logger.exception(f"Fatal: {e}")
import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import os
import logging
import threading

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SCALPING_BOT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración SCALPING M15
SYMBOL = "XAUUSD"
INITIAL_BALANCE = 100.0
TIMEFRAME = mt5.TIMEFRAME_M15
LOT_SIZE = 0.01
MAX_DAILY_TRADES = 20
MAX_SPREAD_USD = 1.0
MIN_ATR_USD = 0.8

# SL/TP para scalping
SL_USD = 1.5
TP_USD = 3.0

simulated_balance = INITIAL_BALANCE


class ScalpingBot:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError("No se pudo inicializar MT5")
        self.trades_today = 0
        self.last_trade_time = None
        self.last_reset = datetime.now().date()
        self.state_file = "bot_state.json"
        self.active_orders = {}  # Rastrear órdenes activas
        self._load_state()
        logger.info(f"Bot inicializado | Capital: {self._get_balance():.2f} USD")

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.trades_today = data.get('trades', 0)
        except:
            pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                'date': str(datetime.now().date()),
                'trades': self.trades_today
            }, f)

    def _reset_daily(self):
        global simulated_balance
        today = datetime.now().date()
        if today != self.last_reset:
            self.trades_today = 0
            self.last_reset = today
            self._save_state()
            logger.info(f"=== NUEVO DÍA === Balance: {simulated_balance:.2f} USD")

    def _get_balance(self):
        global simulated_balance
        return simulated_balance

    def _update_balance(self, new_balance):
        global simulated_balance
        simulated_balance = new_balance
        with open('simulated_balance.json', 'w') as f:
            json.dump({
                'balance': round(simulated_balance, 2),
                'date': str(datetime.now().date())
            }, f)

    def _get_ema(self, series, period):
        if len(series) < period:
            return None
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]

    def _get_rsi(self, series, period=14):
        if len(series) < period:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _get_atr(self, rates, period=14):
        if rates is None or len(rates) < period:
            return None
        df = pd.DataFrame(rates)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        atr = df['tr'].rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else None

    def _get_bollinger_bands(self, series, period=20, std_dev=2):
        if len(series) < period:
            return None, None, None
        bb_middle = series.rolling(period).mean()
        bb_std = series.rolling(period).std()
        bb_upper = bb_middle + (std_dev * bb_std)
        bb_lower = bb_middle - (std_dev * bb_std)
        return bb_upper.iloc[-1], bb_middle.iloc[-1], bb_lower.iloc[-1]

    def _check_signal(self):
        """Verifica señal de scalping sin bloquear"""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
        if rates is None or len(rates) < 20:
            return None, None

        df = pd.DataFrame(rates)
        close = df['close']
        high = df['high']
        low = df['low']

        ema5 = self._get_ema(close, 5)
        ema10 = self._get_ema(close, 10)
        rsi = self._get_rsi(close, 14)
        bb_upper, bb_middle, bb_lower = self._get_bollinger_bands(close, 20, 2)

        if ema5 is None or ema10 is None or rsi is None:
            return None, None

        last_close = close.iloc[-1]
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]

        # COMPRA
        if ema5 > ema10:
            if 40 < rsi < 70:
                if last_low < bb_lower and last_close > bb_lower:
                    return 'buy', 0.75
                elif last_close < bb_middle and rsi < 60:
                    return 'buy', 0.65

        # VENTA
        elif ema5 < ema10:
            if 30 < rsi < 60:
                if last_high > bb_upper and last_close < bb_upper:
                    return 'sell', 0.75
                elif last_close > bb_middle and rsi > 40:
                    return 'sell', 0.65

        return None, None

    def _place_order(self, direction):
        """Coloca orden sin monitoreo (MT5 cierra automáticamente con TP/SL)"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if not symbol_info:
            logger.error("Símbolo no encontrado")
            return None

        point = symbol_info.point
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return None

        sl_points = int(round(SL_USD / 0.01))
        tp_points = int(round(TP_USD / 0.01))

        if direction == 'buy':
            price = tick.ask
            sl_price = price - sl_points * point
            tp_price = price + tp_points * point
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl_price = price + sl_points * point
            tp_price = price - tp_points * point
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 30,
            "magic": 99999,
            "comment": f"SCALP-{direction.upper()}"
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Error orden: {mt5.last_error()}")
            return None

        logger.info(f"🔷 {direction.upper()} {LOT_SIZE} @ {price:.2f} | SL:{SL_USD:.1f} TP:{TP_USD:.1f}")
        return result

    def _monitor_orders_async(self, ticket):
        """Monitorea orden en thread separado SIN BLOQUEAR el loop principal"""
        def monitor():
            monitor_start = time.time()
            max_time = 600

            while time.time() - monitor_start < max_time:
                pos = mt5.positions_get(ticket=ticket)
                if not pos:
                    # Orden cerrada - verificar P&L
                    deals = mt5.history_deals_get(ticket=ticket)
                    if deals:
                        d = deals[-1]
                        pnl = d.profit
                        new_balance = self._get_balance() + pnl
                        self._update_balance(new_balance)

                        if pnl > 0:
                            logger.info(f"✓ +{pnl:.2f} USD | Balance: {new_balance:.2f} USD")
                        else:
                            logger.info(f"✗ {pnl:.2f} USD | Balance: {new_balance:.2f} USD")
                    break

                time.sleep(5)

        # Thread daemon no bloquea el programa principal
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def run(self):
        """Loop principal - NUNCA se congela"""
        logger.info("🚀 SCALPING BOT M15 - NO-BLOCKING MODE")
        logger.info(f"Config: SL={SL_USD} | TP={TP_USD} | Max={MAX_DAILY_TRADES} ops/día")

        account = mt5.account_info()
        if account:
            logger.info(f"Cuenta: {account.name} | Saldo Real: {account.balance:.2f}")

        while True:
            try:
                self._reset_daily()

                if self.trades_today >= MAX_DAILY_TRADES:
                    logger.info(f"Límite {MAX_DAILY_TRADES} ops alcanzado - esperando próximo día")
                    time.sleep(3600)
                    continue

                # Mínimo entre operaciones
                if self.last_trade_time:
                    elapsed = (datetime.now() - self.last_trade_time).total_seconds()
                    if elapsed < 60:
                        time.sleep(10)
                        continue

                # Verificar spread
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    time.sleep(30)
                    continue

                spread_usd = (tick.ask - tick.bid) * 1.0
                if spread_usd > MAX_SPREAD_USD:
                    time.sleep(60)
                    continue

                # Verificar ATR
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
                if rates is None:
                    time.sleep(30)
                    continue

                atr = self._get_atr(rates, 14)
                if atr is None or atr < MIN_ATR_USD:
                    time.sleep(60)
                    continue

                # Buscar señal
                direction, confidence = self._check_signal()
                if direction is None:
                    time.sleep(60)
                    continue

                logger.info(f"📊 Señal: {direction.upper()} (conf: {confidence:.2f})")

                # Colocar orden (MT5 cierra automáticamente con TP/SL)
                result = self._place_order(direction)
                if result:
                    self.trades_today += 1
                    self.last_trade_time = datetime.now()
                    self._save_state()
                    
                    # Monitoreo en thread separado (NO BLOQUEA)
                    self._monitor_orders_async(result.order)
                    logger.info("✅ Orden colocada | Bot continuando con próximas señales...")

                # Espera corta antes de buscar siguiente señal
                time.sleep(15)

            except Exception as e:
                logger.exception(f"Error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    try:
        bot = ScalpingBot()
        bot.run()
    except Exception as e:
        logger.exception(f"Fatal: {e}")
import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import os
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SCALPING_BOT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración SCALPING M15
SYMBOL = "XAUUSD"
INITIAL_BALANCE = 100.0
TIMEFRAME = mt5.TIMEFRAME_M15  # 15 minutos para scalping
LOT_SIZE = 0.01
MAX_DAILY_TRADES = 20  # Hasta 20 ops/día
MAX_SPREAD_USD = 1.0
MIN_ATR_USD = 0.8  # ATR más bajo para M15

# SL/TP para scalping
SL_USD = 1.5  # Stop loss pequeño
TP_USD = 3.0  # Take profit pequeño (ratio 1:2)

# Balance simulado global
simulated_balance = INITIAL_BALANCE


class ScalpingBot:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError("No se pudo inicializar MT5")
        self.trades_today = 0
        self.last_trade_time = None
        self.last_reset = datetime.now().date()
        self.state_file = "bot_state.json"
        self._load_state()
        logger.info(f"Bot inicializado | Capital: {self._get_balance():.2f} USD")

    def _load_state(self):
        """Carga estado del día anterior"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.trades_today = data.get('trades', 0)
        except:
            pass

    def _save_state(self):
        """Guarda estado"""
        with open(self.state_file, 'w') as f:
            json.dump({
                'date': str(datetime.now().date()),
                'trades': self.trades_today
            }, f)

    def _reset_daily(self):
        """Reset diario"""
        global simulated_balance
        today = datetime.now().date()
        if today != self.last_reset:
            self.trades_today = 0
            self.last_reset = today
            self._save_state()
            logger.info(f"=== NUEVO DÍA === Balance: {simulated_balance:.2f} USD")

    def _get_balance(self):
        global simulated_balance
        return simulated_balance

    def _update_balance(self, new_balance):
        """Actualiza balance simulado"""
        global simulated_balance
        simulated_balance = new_balance
        with open('simulated_balance.json', 'w') as f:
            json.dump({
                'balance': round(simulated_balance, 2),
                'date': str(datetime.now().date())
            }, f)

    def _get_ema(self, series, period):
        """Calcula EMA"""
        if len(series) < period:
            return None
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]

    def _get_rsi(self, series, period=14):
        """Calcula RSI"""
        if len(series) < period:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _get_atr(self, rates, period=14):
        """Calcula ATR en USD"""
        if rates is None or len(rates) < period:
            return None
        df = pd.DataFrame(rates)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        atr = df['tr'].rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else None

    def _get_bollinger_bands(self, series, period=20, std_dev=2):
        """Calcula Bollinger Bands"""
        if len(series) < period:
            return None, None, None
        bb_middle = series.rolling(period).mean()
        bb_std = series.rolling(period).std()
        bb_upper = bb_middle + (std_dev * bb_std)
        bb_lower = bb_middle - (std_dev * bb_std)
        return bb_upper.iloc[-1], bb_middle.iloc[-1], bb_lower.iloc[-1]

    def _check_signal(self):
        """
        Verifica señal de scalping:
        1. Cruce de EMAs 5/10
        2. Confirmación RSI
        3. Rebote de Bollinger
        """
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
        if rates is None or len(rates) < 20:
            return None, None

        df = pd.DataFrame(rates)
        close = df['close']
        high = df['high']
        low = df['low']

        # Calcular indicadores
        ema5 = self._get_ema(close, 5)
        ema10 = self._get_ema(close, 10)
        rsi = self._get_rsi(close, 14)
        bb_upper, bb_middle, bb_lower = self._get_bollinger_bands(close, 20, 2)

        if ema5 is None or ema10 is None or rsi is None:
            return None, None

        last_close = close.iloc[-1]
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]

        # SEÑAL COMPRA
        if ema5 > ema10:  # EMA 5 encima de EMA 10 (tendencia alcista)
            if 40 < rsi < 70:  # RSI confirmación alcista
                # Rebote de banda inferior = mayor confianza
                if last_low < bb_lower and last_close > bb_lower:
                    return 'buy', 0.75
                # O simplemente cerca de BB media
                elif last_close < bb_middle and rsi < 60:
                    return 'buy', 0.65

        # SEÑAL VENTA
        elif ema5 < ema10:  # EMA 5 debajo de EMA 10 (tendencia bajista)
            if 30 < rsi < 60:  # RSI confirmación bajista
                # Rebote de banda superior = mayor confianza
                if last_high > bb_upper and last_close < bb_upper:
                    return 'sell', 0.75
                # O simplemente cerca de BB media
                elif last_close > bb_middle and rsi > 40:
                    return 'sell', 0.65

        return None, None

    def _place_order(self, direction):
        """Coloca orden de scalping"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if not symbol_info:
            logger.error("Símbolo no encontrado")
            return None

        point = symbol_info.point
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return None

        # Convertir SL/TP USD a puntos
        sl_points = int(round(SL_USD / 0.01))
        tp_points = int(round(TP_USD / 0.01))

        if direction == 'buy':
            price = tick.ask
            sl_price = price - sl_points * point
            tp_price = price + tp_points * point
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl_price = price + sl_points * point
            tp_price = price - tp_points * point
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 30,
            "magic": 99999,
            "comment": f"SCALP-{direction.upper()}-SL{SL_USD:.1f}-TP{TP_USD:.1f}"
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Error orden: {mt5.last_error()}")
            return None

        logger.info(f"🔷 {direction.upper()} {LOT_SIZE} @ {price:.2f} | SL:{SL_USD:.1f} TP:{TP_USD:.1f}")
        return result

    def _monitor_trade(self, ticket):
        """Monitorea operación hasta cierre"""
        monitor_start = time.time()
        max_monitor_time = 600  # 10 minutos máximo

        while time.time() - monitor_start < max_monitor_time:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                # Cerrada
                deals = mt5.history_deals_get(ticket=ticket)
                if deals:
                    d = deals[-1]
                    pnl = d.profit
                    new_balance = self._get_balance() + pnl
                    self._update_balance(new_balance)

                    if pnl > 0:
                        logger.info(f"✓ +{pnl:.2f} USD | Balance: {new_balance:.2f} USD")
                    else:
                        logger.info(f"✗ {pnl:.2f} USD | Balance: {new_balance:.2f} USD")
                break

            time.sleep(5)

    def run(self):
        """Loop principal de scalping"""
        logger.info("🚀 SCALPING BOT INICIADO - M15")
        logger.info(f"Config: SL={SL_USD} | TP={TP_USD} | Max={MAX_DAILY_TRADES} ops/día")

        account = mt5.account_info()
        if account:
            logger.info(f"Cuenta: {account.name} | Saldo Real: {account.balance:.2f}")

        while True:
            try:
                self._reset_daily()

                if self.trades_today >= MAX_DAILY_TRADES:
                    logger.info(f"Límite {MAX_DAILY_TRADES} ops alcanzado")
                    time.sleep(3600)
                    continue

                # Esperar mínimo entre operaciones
                if self.last_trade_time:
                    elapsed = (datetime.now() - self.last_trade_time).total_seconds()
                    if elapsed < 60:  # Mínimo 1 min entre ops
                        time.sleep(10)
                        continue

                # Verificar spread
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    time.sleep(30)
                    continue

                spread_usd = (tick.ask - tick.bid) * 1.0
                if spread_usd > MAX_SPREAD_USD:
                    logger.debug(f"Spread alto {spread_usd:.2f}")
                    time.sleep(60)
                    continue

                # Verificar ATR
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
                if rates is None:
                    time.sleep(30)
                    continue

                atr = self._get_atr(rates, 14)
                if atr is None or atr < MIN_ATR_USD:
                    time.sleep(60)
                    continue

                # Buscar señal
                direction, confidence = self._check_signal()
                if direction is None:
                    time.sleep(60)
                    continue

                logger.info(f"Señal: {direction.upper()} (conf: {confidence:.2f})")

                # Colocar orden
                result = self._place_order(direction)
                if result:
                    self.trades_today += 1
                    self.last_trade_time = datetime.now()
                    self._save_state()
                    # Monitorear en background
                    self._monitor_trade(result.order)

                time.sleep(30)

            except Exception as e:
                logger.exception(f"Error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    try:
        bot = ScalpingBot()
        bot.run()
    except Exception as e:
        logger.exception(f"Fatal: {e}")
import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import os
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - XAUUSD_BOT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración
SYMBOL = "XAUUSD"
INITIAL_BALANCE = 100.0
MIN_CONFIDENCE = 0.60
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
TIMEFRAME_H4 = mt5.TIMEFRAME_H4

# Gestión de riesgo
LOT_SIZE = 0.01
MAX_DAILY_TRADES = 10
MAX_DRAWDOWN_PERCENT = 40
TRAILING_TRIGGER_PROFIT = 8.0
MIN_ATR_USD = 1.2
MAX_SPREAD_USD = 1.5
MIN_MOVE_TO_TRADE_USD = 2.0

# Estrategia
USE_HOUR_FILTER = False
ACTIVE_HOURS_START = 8
ACTIVE_HOURS_END = 22

# Balance simulado
simulated_balance = INITIAL_BALANCE


class MidasOrquestador:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError("No se pudo inicializar MT5")
        self.trades_today = 0
        self.last_trade_time = None
        self.last_reset = datetime.now().date()
        self.state_file = "bot_state.json"
        self._load_state()

    def _load_state(self):
        global simulated_balance
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.trades_today = data.get('trades', 0)
        except:
            pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                'date': str(datetime.now().date()),
                'trades': self.trades_today
            }, f)

    def _reset_daily(self):
        global simulated_balance
        today = datetime.now().date()
        if today != self.last_reset:
            self.trades_today = 0
            self.last_reset = today
            self._save_state()
            logger.info(f"Nuevo día: balance {simulated_balance:.2f} USD")

    def _get_balance(self):
        global simulated_balance
        return simulated_balance

    def _update_balance(self, new_balance):
        global simulated_balance
        simulated_balance = new_balance
        with open('simulated_balance.json', 'w') as f:
            json.dump({'balance': round(simulated_balance, 2), 'date': str(datetime.now().date())}, f)

    def _hour_filter(self):
        if not USE_HOUR_FILTER:
            return True
        now_hour = datetime.now().hour
        return ACTIVE_HOURS_START <= now_hour < ACTIVE_HOURS_END

    def _get_atr(self, rates, period=14):
        """Calcula ATR en USD"""
        if rates is None or len(rates) < period:
            return None
        df = pd.DataFrame(rates)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        atr = df['tr'].rolling(period).mean().iloc[-1]
        return atr * 1.0 if not pd.isna(atr) else None

    def _get_rsi(self, series, period=14):
        """Calcula RSI"""
        if len(series) < period:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

    def _get_ema(self, series, period):
        """Calcula EMA"""
        if len(series) < period:
            return None
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]

    def _get_bollinger_bands(self, series, period=20, std_dev=2):
        """Calcula Bollinger Bands"""
        if len(series) < period:
            return None, None, None
        bb_middle = series.rolling(period).mean()
        bb_std = series.rolling(period).std()
        bb_upper = bb_middle + (std_dev * bb_std)
        bb_lower = bb_middle - (std_dev * bb_std)
        return bb_upper.iloc[-1], bb_middle.iloc[-1], bb_lower.iloc[-1]

    def _check_trend_h4(self):
        """Verifica tendencia en H4 usando EMA 20/50"""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_H4, 0, 100)
        if rates is None or len(rates) < 50:
            return None
        
        df = pd.DataFrame(rates)
        close = df['close']
        
        ema20 = self._get_ema(close, 20)
        ema50 = self._get_ema(close, 50)
        last_close = close.iloc[-1]
        
        if ema20 is None or ema50 is None:
            return None
        
        # Tendencia clara: precio por encima/debajo de EMAs
        if last_close > ema20 > ema50:
            return 'buy'
        elif last_close < ema20 < ema50:
            return 'sell'
        return None

    def _check_entry_signal_h1(self, direction):
        """Verifica señal de entrada en H1"""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_H1, 0, 50)
        if rates is None or len(rates) < 20:
            return False, None
        
        df = pd.DataFrame(rates)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 1. RSI en zona neutral (40-60) - evita extremos
        rsi = self._get_rsi(close, 14)
        if rsi is None or rsi < 40 or rsi > 60:
            return False, None
        
        # 2. Confirmar con Bollinger Bands (rebote de bandas)
        bb_upper, bb_middle, bb_lower = self._get_bollinger_bands(close, 20, 2)
        if bb_upper is None:
            return False, None
        
        last_close = close.iloc[-1]
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]
        
        if direction == 'buy':
            # Señal: precio rebotó de BB inferior (bullish reversal)
            if last_low < bb_lower * 1.001 and last_close > bb_lower:
                confidence = 0.70
                return True, confidence
            # O: precio en BB inferior + RSI bajo pero > 40
            if last_close < bb_middle and rsi < 50:
                confidence = 0.65
                return True, confidence
        
        elif direction == 'sell':
            # Señal: precio rebotó de BB superior (bearish reversal)
            if last_high > bb_upper * 0.999 and last_close < bb_upper:
                confidence = 0.70
                return True, confidence
            # O: precio en BB superior + RSI alto pero < 60
            if last_close > bb_middle and rsi > 50:
                confidence = 0.65
                return True, confidence
        
        return False, None

    def _calculate_sl_tp(self, atr, direction):
        """Calcula SL y TP basado en ATR"""
        if atr is None or atr < MIN_ATR_USD:
            return None, None
        
        # SL = 1.5 * ATR
        sl_usd = atr * 1.5
        
        # TP = 3 * ATR (ratio 1:2)
        tp_usd = atr * 3.0
        
        # Mínimos
        if tp_usd < MIN_MOVE_TO_TRADE_USD:
            return None, None
        
        return sl_usd, tp_usd

    def _place_order(self, direction, sl_usd, tp_usd):
        """Coloca orden"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if not symbol_info:
            logger.error("Símbolo no encontrado")
            return None
        
        point = symbol_info.point
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error("No se pudo obtener tick")
            return None
        
        sl_points = int(round(sl_usd / 0.01))
        tp_points = int(round(tp_usd / 0.01))
        
        if direction == 'buy':
            price = tick.ask
            sl_price = price - sl_points * point
            tp_price = price + tp_points * point
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl_price = price + sl_points * point
            tp_price = price - tp_points * point
            order_type = mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 12345,
            "comment": f"XAUUSD {direction.upper()} SL:{sl_usd:.2f} TP:{tp_usd:.2f}"
        }
        
        result = mt5.order_send(request)
        if result is None:
            logger.error("order_send retornó None")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Error orden: {mt5.last_error()} (retcode {result.retcode})")
            return None
        
        logger.info(f"Orden {direction.upper()} {LOT_SIZE} @ {price:.2f} | SL:{sl_usd:.2f} TP:{tp_usd:.2f}")
        return result

    def _monitor_trade(self, order_ticket):
        """Monitorea operación abierta"""
        start_time = time.time()
        trailing_activated = False
        
        while time.time() - start_time < 3600:
            pos = mt5.positions_get(ticket=order_ticket)
            if not pos:
                # Operación cerrada
                deal_info = mt5.history_deals_get(ticket=order_ticket)
                if deal_info:
                    d = deal_info[-1]
                    pnl = d.profit
                    if pnl > 0:
                        self._update_balance(self._get_balance() + pnl)
                        logger.info(f"✓ GANANCIA: +{pnl:.2f} USD | Balance: {self._get_balance():.2f} USD")
                    else:
                        self._update_balance(self._get_balance() + pnl)
                        logger.info(f"✗ PÉRDIDA: {pnl:.2f} USD | Balance: {self._get_balance():.2f} USD")
                break
            
            pos = pos[0]
            current_profit = pos.profit
            
            # Trailing stop: activar si ganancia > 8 USD
            if current_profit >= TRAILING_TRIGGER_PROFIT and not trailing_activated:
                trailing_activated = True
                logger.info(f"Trailing stop activado (ganancia {current_profit:.2f})")
            
            # Trailing stop: mover SL 50% de ganancia
            if trailing_activated and current_profit > 0:
                new_sl = pos.price_open + (current_profit * 0.5)
                if pos.type == mt5.ORDER_TYPE_BUY:
                    new_sl = max(pos.sl, new_sl - pos.price_open)
                
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": SYMBOL,
                    "position": order_ticket,
                    "sl": new_sl,
                    "tp": pos.tp
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SL actualizado a {new_sl:.5f}")
            
            time.sleep(30)

    def run(self):
        """Loop principal"""
        logger.info(f"Bot iniciado - Capital {self._get_balance():.2f} USD")
        logger.info("Estrategia: Técnica pura (Tendencia H4 + Entrada H1 + RSI/BB)")
        
        account = mt5.account_info()
        if account:
            logger.info(f"Conectado a DEMO | Saldo real: {account.balance:.2f}")
        
        while True:
            try:
                self._reset_daily()
                
                if self.trades_today >= MAX_DAILY_TRADES:
                    logger.info(f"Límite de {MAX_DAILY_TRADES} operaciones alcanzado hoy")
                    time.sleep(3600)
                    continue
                
                if not self._hour_filter():
                    time.sleep(300)
                    continue
                
                if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < 60:
                    time.sleep(30)
                    continue
                
                # Paso 1: Verificar tendencia H4
                trend = self._check_trend_h4()
                if trend is None:
                    logger.info("Sin tendencia clara en H4")
                    time.sleep(60)
                    continue
                
                logger.info(f"Tendencia {trend.upper()} en H4")
                
                # Paso 2: Verificar señal de entrada H1
                signal_valid, confidence = self._check_entry_signal_h1(trend)
                if not signal_valid:
                    time.sleep(60)
                    continue
                
                logger.info(f"Señal {trend.upper()} con confianza {confidence:.2f}")
                
                # Paso 3: Verificar spread
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    time.sleep(30)
                    continue
                
                spread_usd = (tick.ask - tick.bid) * 1.0
                if spread_usd > MAX_SPREAD_USD:
                    logger.info(f"Spread alto {spread_usd:.2f} > {MAX_SPREAD_USD}")
                    time.sleep(60)
                    continue
                
                # Paso 4: Calcular SL/TP
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_H1, 0, 50)
                if rates is None:
                    time.sleep(30)
                    continue
                
                atr = self._get_atr(rates, 14)
                if atr is None or atr < MIN_ATR_USD:
                    logger.info(f"ATR bajo {atr}")
                    time.sleep(60)
                    continue
                
                sl_usd, tp_usd = self._calculate_sl_tp(atr, trend)
                if sl_usd is None:
                    time.sleep(30)
                    continue
                
                # Verificar que TP neto sea positivo
                net_tp = tp_usd - spread_usd
                if net_tp <= 0:
                    logger.info(f"TP neto {net_tp:.2f} no es positivo")
                    time.sleep(30)
                    continue
                
                # Paso 5: Colocar orden
                result = self._place_order(trend, sl_usd, tp_usd)
                if result:
                    self.trades_today += 1
                    self.last_trade_time = datetime.now()
                    self._save_state()
                    
                    # Monitorear operación
                    self._monitor_trade(result.order)
                
                time.sleep(60)
                
            except Exception as e:
                logger.exception(f"Error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    try:
        orq = MidasOrquestador()
        orq.run()
    except Exception as e:
        logger.exception(f"Error fatal: {e}")
import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import os
from config_ai import *
from data_vision import DataVision
from brain_model import BrainModel
from mt5_execution import MT5Executor

class MidasOrquestador:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError("No se pudo inicializar MT5")
        self.dv = DataVision()
        self.brain = BrainModel()
        self.executor = MT5Executor()
        self.losses_today = 0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.last_reset = datetime.now().date()
        self.state_file = "bot_state.json"
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.losses_today = data.get('losses', 0)
                        self.trades_today = data.get('trades', 0)
                        self.consecutive_losses = data.get('consecutive_losses', 0)
            except:
                pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                'date': str(datetime.now().date()),
                'losses': self.losses_today,
                'trades': self.trades_today,
                'consecutive_losses': self.consecutive_losses
            }, f)

    def _reset_daily(self):
        today = datetime.now().date()
        if today != self.last_reset:
            self.losses_today = 0
            self.trades_today = 0
            self.consecutive_losses = 0
            self.last_reset = today
            self._save_state()
            logger.info("Nuevo día: reinicio de contadores")

    def _hour_filter(self):
        if not USE_HOUR_FILTER:
            return True
        now_hour = datetime.now().hour
        return ACTIVE_HOURS_START <= now_hour < ACTIVE_HOURS_END

    def _get_trend(self):
        # Filtro de tendencia muy suave (casi no filtra)
        tf = getattr(mt5, "TIMEFRAME_H4")
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 200)
        if rates is None or len(rates) < 200:
            return None
        df = pd.DataFrame(rates)
        df['ema200'] = df['close'].rolling(200).mean()
        last_close = df['close'].iloc[-1]
        last_ema = df['ema200'].iloc[-1]
        if pd.isna(last_ema):
            return None
        # Umbral muy amplio (2%) para no filtrar casi nada
        if last_close > last_ema * 1.02:
            return 'buy'
        elif last_close < last_ema * 0.98:
            return 'sell'
        return None

    def _check_momentum(self, direction):
        # Filtro de momentum activado - evita extremos de RSI
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 30)
        if rates is None or len(rates) < 14:
            return True
        df = pd.DataFrame(rates)
        rsi = self._calculate_rsi(df['close'], 14)
        last_rsi = rsi.iloc[-1]
        
        if direction == 'buy' and last_rsi > 70:
            logger.info(f"RSI sobrecomprado ({last_rsi:.1f} > 70), evitando COMPRA")
            return False
        elif direction == 'sell' and last_rsi < 30:
            logger.info(f"RSI sobrevendido ({last_rsi:.1f} < 30), evitando VENTA")
            return False
        return True
    
    @staticmethod
    def _calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _get_latest_features(self):
        rates = mt5.copy_rates_from_pos(SYMBOL, self.dv.timeframe, 0, 150)
        if rates is None or len(rates) < 100:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = self.dv.prepare_features_for_live(df)
        return df.iloc[-1:].copy()

    def run(self):
        if not self.executor.connect():
            return
        try:
            self.dv.activate_symbol()
        except Exception as e:
            logger.error(f"No se pudo activar símbolo: {e}")
            return

        if not self.brain.load():
            logger.info("Entrenando nuevo modelo...")
            df_hist = self.dv.download_historical_data(years=3)
            if df_hist is None or len(df_hist) < 1000:
                logger.error("Datos históricos insuficientes")
                return
            X, y, feature_cols = self.dv.prepare_features(df_hist)
            if X is None or len(X) == 0:
                logger.error("No se pudieron generar features")
                return
            self.brain.feature_cols = feature_cols
            self.brain.train(X, y)
            self.brain.save()
        else:
            logger.info("Modelo cargado")

        logger.info(f"Bot iniciado - Capital {self.executor.get_current_balance():.2f} USD")
        logger.info(f"Confianza mínima: {MIN_CONFIDENCE}, ATR mínimo: {MIN_ATR_USD}, TP mínimo: {MIN_MOVE_TO_TRADE_USD}")
        while True:
            try:
                self._reset_daily()

                if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    pause_until = datetime.now() + timedelta(hours=CONSECUTIVE_LOSS_PAUSE_HOURS)
                    logger.warning(f"{self.consecutive_losses} pérdidas consecutivas. Pausa hasta {pause_until.strftime('%H:%M')}")
                    time.sleep(CONSECUTIVE_LOSS_PAUSE_HOURS * 3600)
                    self.consecutive_losses = 0
                    self._save_state()
                    continue

                if self.losses_today >= MAX_DAILY_LOSSES:
                    logger.warning(f"{MAX_DAILY_LOSSES} pérdidas hoy. Pausa hasta mañana.")
                    time.sleep(3600)
                    continue
                if self.trades_today >= MAX_DAILY_TRADES:
                    logger.info(f"Límite de {MAX_DAILY_TRADES} operaciones alcanzado.")
                    time.sleep(3600)
                    continue
                if not self._hour_filter():
                    time.sleep(60)
                    continue
                if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < 30:
                    time.sleep(5)
                    continue

                latest = self._get_latest_features()
                if latest is None:
                    time.sleep(10)
                    continue
                if self.brain.feature_cols is None:
                    time.sleep(10)
                    continue
                missing = [c for c in self.brain.feature_cols if c not in latest.columns]
                if missing:
                    for c in missing:
                        latest[c] = 0.0
                X_pred = latest[self.brain.feature_cols]
                pred, conf, prob_up = self.brain.predict(X_pred)
                direction = 'buy' if pred == 1 else 'sell'
                logger.info(f"Señal: {direction} | confianza {conf:.2f} | prob_subida {prob_up:.2f}")

                if conf < MIN_CONFIDENCE:
                    logger.info(f"Confianza baja ({conf:.2f} < {MIN_CONFIDENCE}), esperando...")
                    time.sleep(10)
                    continue

                # Filtro de tendencia (muy suave)
                trend = self._get_trend()
                if trend is not None and trend != direction:
                    logger.info(f"Señal {direction} contra tendencia {trend}, omitida")
                    time.sleep(60)
                    continue

                # Filtro de momentum (RSI) - evita extremos
                if not self._check_momentum(direction):
                    time.sleep(60)
                    continue

                spread_usd = self.executor.get_current_spread_usd()
                if spread_usd > MAX_SPREAD_USD:
                    logger.info(f"Spread alto {spread_usd:.2f} > {MAX_SPREAD_USD}")
                    time.sleep(30)
                    continue

                atr_val = latest['atr'].values[0] if 'atr' in latest else 1.5
                if atr_val < MIN_ATR_USD:
                    logger.info(f"ATR bajo {atr_val:.2f} < {MIN_ATR_USD}")
                    time.sleep(60)
                    continue

                sl_usd, tp_usd = self.brain.dynamic_tp_sl(atr_val, conf, direction)
                if tp_usd < MIN_MOVE_TO_TRADE_USD:
                    logger.info(f"TP {tp_usd:.2f} < mínimo {MIN_MOVE_TO_TRADE_USD}")
                    time.sleep(30)
                    continue

                net_tp = tp_usd - spread_usd
                if net_tp <= 0:
                    logger.info(f"TP neto {net_tp:.2f} no es positivo")
                    time.sleep(30)
                    continue

                result = self.executor.place_order(direction, sl_usd, tp_usd, conf)
                if result:
                    self.trades_today += 1
                    self.last_trade_time = datetime.now()
                    self._save_state()
                    self._monitor_trade(result.order)
                time.sleep(30)
            except Exception as e:
                logger.exception(e)
                time.sleep(30)

    def _monitor_trade(self, ticket):
        start_time = time.time()
        trailing_activated = False
        entry_price = None
        direction = None

        pos = mt5.positions_get(ticket=ticket)
        if pos:
            pos = pos[0]
            entry_price = pos.price_open
            direction = 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell'
        else:
            return

        while True:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                break
            pos = pos[0]
            profit = pos.profit
            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                time.sleep(1)
                continue
            current_price = tick.bid if direction == 'buy' else tick.ask

            # Trailing stop muy agresivo
            if not trailing_activated and profit >= TRAILING_TRIGGER_PROFIT:
                if direction == 'buy':
                    new_sl = entry_price + TRAILING_DISTANCE_USD
                else:
                    new_sl = entry_price - TRAILING_DISTANCE_USD
                if self.executor.modify_sl_tp(ticket, new_sl):
                    trailing_activated = True
                    logger.info(f"Trailing activado: SL a {new_sl:.2f} (ganancia asegurada +{TRAILING_DISTANCE_USD} USD)")
            elif trailing_activated:
                if direction == 'buy':
                    new_sl = current_price - TRAILING_DISTANCE_USD
                    if new_sl > pos.sl:
                        if self.executor.modify_sl_tp(ticket, new_sl):
                            logger.info(f"Trailing: SL a {new_sl:.2f} (dist {TRAILING_DISTANCE_USD} USD)")
                else:
                    new_sl = current_price + TRAILING_DISTANCE_USD
                    if new_sl < pos.sl:
                        if self.executor.modify_sl_tp(ticket, new_sl):
                            logger.info(f"Trailing: SL a {new_sl:.2f} (dist {TRAILING_DISTANCE_USD} USD)")

            if time.time() - start_time > 7200:
                logger.warning(f"Posición {ticket} abierta >2h, cerrando")
                self.executor.close_position(ticket)
                break
            time.sleep(2)

        history = mt5.history_deals_get(position=ticket)
        if history:
            net_profit = sum(d.profit for d in history)
            if not USE_REAL_BALANCE:
                current_bal = self.executor.get_current_balance()
                new_bal = current_bal + net_profit
                self.executor.update_simulated_balance(new_bal)
            if net_profit < 0:
                self.losses_today += 1
                self.consecutive_losses += 1
                self._save_state()
                logger.warning(f"Pérdida {net_profit:.2f} USD. Pérdidas hoy: {self.losses_today}, consecutivas: {self.consecutive_losses}")
            else:
                self.consecutive_losses = 0
                self._save_state()
                logger.info(f"Ganancia {net_profit:.2f} USD. Racha reiniciada.")
        else:
            logger.warning(f"No se encontró historial para ticket {ticket}")

if __name__ == "__main__":
    bot = MidasOrquestador()
    bot.run()