import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config_ai import MODEL_PATH, FEATURES_PATH, logger, MIN_CONFIDENCE, MIN_SL_USD, MAX_SL_USD, MIN_RISK_REWARD_RATIO, MAX_RISK_REWARD_RATIO, ATR_MULTIPLIER_SL

class BrainModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.cv_score = 0.0

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info(f"Entrenando modelo XGBoost con {len(X)} muestras...")
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(
                n_estimators=1000,          # Aumentado de 500 - mejor aprendizaje
                max_depth=6,                # Aumentado de 4 - capas mas profundas
                learning_rate=0.08,         # Aumentado de 0.04 - aprendizaje mas rapido
                subsample=0.85,             # Aumentado de 0.8 - mas datos por iteracion
                colsample_bytree=0.85,      # Aumentado de 0.8 - mas features
                min_child_weight=3,         # Reducido de 5 - mas flexible
                gamma=2.0,                  # Nuevo - regularizacion L1
                scale_pos_weight=(len(y)-y.sum())/y.sum(),
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ))
        ])
        self.model.fit(X, y)
        acc = self.model.score(X, y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = cross_val_score(self.model, X, y, cv=skf, scoring='f1')
        self.cv_score = f1_scores.mean()
        logger.info(f"Precisión train: {acc:.2f} | F1 validación: {self.cv_score:.2f} (+/- {f1_scores.std():.2f})")
        self.feature_cols = list(X.columns)

    def predict(self, X: pd.DataFrame):
        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]
        confidence = max(proba[0], proba[1])
        prob_up = proba[1]
        return pred, confidence, prob_up

    def dynamic_tp_sl(self, atr_usd: float, confidence: float, direction: str):
        sl_usd = atr_usd * ATR_MULTIPLIER_SL
        # Invertido: confianza ALTA reduce SL, confianza BAJA aumenta SL (conservador)
        confidence_factor = 1.0 / (0.7 + (1 - confidence) * 0.5)  # [0.77 a 1.43]
        sl_usd = sl_usd * confidence_factor
        sl_usd = max(MIN_SL_USD, min(MAX_SL_USD, sl_usd))
        ratio = MIN_RISK_REWARD_RATIO + (confidence - MIN_CONFIDENCE) * 2.0
        ratio = max(MIN_RISK_REWARD_RATIO, min(MAX_RISK_REWARD_RATIO, ratio))
        tp_usd = sl_usd * ratio
        return round(sl_usd, 2), round(tp_usd, 2)

    def save(self):
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.feature_cols, FEATURES_PATH)
        logger.info("Modelo XGBoost guardado")

    def load(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.feature_cols = joblib.load(FEATURES_PATH)
            logger.info("Modelo XGBoost cargado correctamente")
            return True
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo: {e}")
            return False