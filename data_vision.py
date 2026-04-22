import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
from config_ai import TIMEFRAME_MAIN, SYMBOL, SPREAD_COST_USD, logger

class DataVision:
    def __init__(self):
        self.timeframe = self._get_timeframe(TIMEFRAME_MAIN)

    def _get_timeframe(self, tf_str):
        mapping = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                   "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
        return mapping.get(tf_str, mt5.TIMEFRAME_H1)

    def activate_symbol(self):
        if not mt5.symbol_select(SYMBOL, True):
            raise Exception(f"No se pudo activar {SYMBOL}")
        for _ in range(10):
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                break
            time.sleep(0.5)
        logger.info(f"Símbolo {SYMBOL} activado")

    def download_historical_data(self, years=3):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        for attempt in range(3):
            rates = mt5.copy_rates_range(SYMBOL, self.timeframe, start_date, end_date)
            if rates is not None and len(rates) > 100:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                logger.info(f"Descargadas {len(df)} velas ({start_date.date()} a {end_date.date()})")
                return df
            else:
                logger.warning(f"Intento {attempt+1} falló. Reintentando...")
                time.sleep(3)
        rates = mt5.copy_rates_from_pos(SYMBOL, self.timeframe, 0, 100000)
        if rates is not None and len(rates) > 1000:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            logger.info(f"Descargadas {len(df)} velas con copy_rates_from_pos")
            return df
        logger.error("No se obtuvieron datos históricos")
        return None

    @staticmethod
    def add_candle_morphology(df):
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['body_pct'] = df['body'] / (df['candle_range'] + 1e-9)
        return df

    def prepare_features(self, df):
        df = df.copy()
        # Target bidireccional con umbral de 0.08% (~4 USD) para movimientos significativos
        future_return = (df['close'].shift(-1) - df['close']) / df['close'] * 100
        df['target'] = 2  # valor por defecto (descartar)
        df.loc[future_return > 0.08, 'target'] = 1   # COMPRA - AUMENTADO de 0.04
        df.loc[future_return < -0.08, 'target'] = 0  # VENTA - AUMENTADO de -0.04
        # Eliminar filas con movimiento pequeño (neutral)
        df = df.dropna()
        df = df[df['target'].isin([0, 1])]
        df = self._add_all_indicators(df)
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in ['target']]
        X = df[feature_cols]
        y = df['target']
        buy_pct = (y == 1).sum() / len(y) * 100
        sell_pct = (y == 0).sum() / len(y) * 100
        logger.info(f"Target: COMPRAR={buy_pct:.1f}% | VENDER={sell_pct:.1f}% (umbral 0.08% - movimientos significativos)")
        return X, y, feature_cols

    def prepare_features_for_live(self, df):
        df = df.copy()
        df = self._add_all_indicators(df)
        df = df.bfill().fillna(0)
        return df

    def _add_all_indicators(self, df):
        df['returns'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(14).std()
        df['atr'] = self._atr(df, 14)
        df['rsi'] = self._rsi(df['close'], 14)
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        bb_upper, bb_lower = self._bollinger(df['close'], 20)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_upper - bb_lower
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-9)
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        df['momentum_5'] = df['close'].pct_change(5)
        df = self.add_candle_morphology(df)
        return df

    @staticmethod
    def _atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _bollinger(series, period=20, std_dev=2):
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, lower