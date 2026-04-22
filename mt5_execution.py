import MetaTrader5 as mt5
import json
import os
from datetime import datetime
from config_ai import *

class MT5Executor:
    def __init__(self):
        self.connected = False
        self.balance_file = BALANCE_FILE
        self.balance = self._load_balance()
        self.initial_balance = self.balance

    def _load_balance(self):
        if USE_REAL_BALANCE:
            return None
        if os.path.exists(self.balance_file):
            try:
                with open(self.balance_file, 'r') as f:
                    data = json.load(f)
                    balance = data.get('balance', SIMULATED_BALANCE)
                    if RESET_BALANCE_DAILY:
                        last_date = data.get('date')
                        if last_date != str(datetime.now().date()):
                            balance = SIMULATED_BALANCE
                    logger.info(f"Balance cargado desde archivo: {balance:.2f} USD")
                    return balance
            except Exception as e:
                logger.warning(f"Error al cargar balance: {e}")
        logger.info(f"Usando balance inicial: {SIMULATED_BALANCE:.2f} USD")
        return SIMULATED_BALANCE

    def _save_balance(self):
        if USE_REAL_BALANCE:
            return
        try:
            with open(self.balance_file, 'w') as f:
                json.dump({
                    'balance': self.balance,
                    'date': str(datetime.now().date())
                }, f)
            logger.debug(f"Balance guardado: {self.balance:.2f} USD")
        except Exception as e:
            logger.error(f"Error al guardar balance: {e}")

    def get_current_balance(self):
        if USE_REAL_BALANCE:
            acc = mt5.account_info()
            return acc.balance if acc else 0.0
        else:
            return self.balance

    def get_drawdown_percent(self):
        current = self.get_current_balance()
        if self.initial_balance <= 0:
            return 0
        return (self.initial_balance - current) / self.initial_balance * 100

    def calculate_dynamic_lot(self, confidence, drawdown_percent=0.0):
        """Calcula lote dinamico para capital bajo ($100 o menos)"""
        # Para capital bajo, usar lote fijo 0.01 (mínimo válido en MT5)
        # La confianza y drawdown solo determinan SI tradear, no el tamaño
        
        # No tradear si drawdown es muy alto
        if drawdown_percent > 40:
            return 0.0
        
        # No tradear si confianza es baja
        if confidence < 0.65:
            return 0.0
        
        # Usar lote mínimo 0.01 para todas las operaciones válidas
        return 0.01

    def update_simulated_balance(self, new_balance):
        if not USE_REAL_BALANCE:
            old = self.balance
            self.balance = new_balance
            self._save_balance()
            logger.info(f"Balance actualizado: {old:.2f} USD -> {new_balance:.2f} USD (drawdown: {self.get_drawdown_percent():.1f}%)")

    def connect(self):
        if not mt5.terminal_info():
            if not mt5.initialize():
                logger.error("No se pudo inicializar MT5")
                return False
        acc = mt5.account_info()
        if acc is None:
            logger.error("No hay cuenta activa")
            return False
        real_balance = acc.balance
        if USE_REAL_BALANCE:
            logger.info(f"Conectado a cuenta REAL. Balance: {real_balance:.2f} USD")
        else:
            logger.info(f"Conectado a DEMO, usando CAPITAL SIMULADO: {self.balance:.2f} USD (ignorando saldo real {real_balance:.2f})")
        self.connected = True
        return True

    def get_current_spread_usd(self):
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return SPREAD_COST_USD
        point = mt5.symbol_info(SYMBOL).point
        spread_points = (tick.ask - tick.bid) / point
        return spread_points * 0.01

    def _usd_to_points(self, usd_amount):
        return int(round(usd_amount / 0.01))

    def place_order(self, direction, sl_usd, tp_usd, confidence):
        if sl_usd > 10.0:
            logger.error(f"SL de {sl_usd} USD supera el límite de 10 USD. Operación cancelada.")
            return None

        if confidence < MIN_CONFIDENCE:
            logger.info(f"Confianza {confidence:.2f} < {MIN_CONFIDENCE}, no se opera")
            return None

        symbol_info = mt5.symbol_info(SYMBOL)
        if not symbol_info:
            logger.error("Símbolo no encontrado")
            return None

        point = symbol_info.point
        sl_points = self._usd_to_points(sl_usd)
        tp_points = self._usd_to_points(tp_usd)
        
        # Usar lote dinámico basado en confianza y drawdown
        drawdown = self.get_drawdown_percent()
        lot_size = self.calculate_dynamic_lot(confidence, drawdown)
        
        if lot_size == 0:
            logger.info(f"Lote dinamico = 0 (confianza baja o drawdown alto)")
            return None

        tick = mt5.symbol_info_tick(SYMBOL)
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
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"ML_{direction}_{confidence:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Error orden: {result.comment} (retcode {result.retcode})")
            return None
        logger.info(f"Orden {direction} ejecutada a {price} | SL: {sl_price} ({sl_usd} USD) | TP: {tp_price} ({tp_usd} USD) | Lote: {lot_size}")
        return result

    def modify_sl_tp(self, ticket, new_sl_price, new_tp_price=None):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl_price,
        }
        if new_tp_price is not None:
            request["tp"] = new_tp_price
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Error modificando SL/TP: {result.comment} (retcode {result.retcode})")
            return False
        logger.info(f"SL modificado a {new_sl_price:.2f} para posición {ticket}")
        return True

    def close_position(self, ticket):
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return False
        close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE