"""
siem — thư viện dùng chung cho pipeline SIEM AI Kill Chain Detection.

Các script 01-04 ở thư mục gốc chỉ orchestrate (gọi các class dưới đây theo
đúng thứ tự); toàn bộ logic nghiệp vụ (load data, feature engineering, model,
session building, risk scoring, evaluation) sống trong package này để không
lặp lại code giữa các bước của pipeline.
"""

from .killchain import KillChain
from .data import AlertDataLoader
from .features import FeatureEngineer
from .model import ModelTrainer, ThresholdedPredictor, ThresholdTuner
from .risk import RiskScorer
from .session import SessionBuilder, SessionThresholdTuner
from .evaluation import SessionEvaluator
from .response import ResponsePlaybook, ResponseExecutor

__all__ = [
    "KillChain",
    "AlertDataLoader",
    "FeatureEngineer",
    "ModelTrainer",
    "ThresholdedPredictor",
    "ThresholdTuner",
    "RiskScorer",
    "SessionBuilder",
    "SessionThresholdTuner",
    "SessionEvaluator",
    "ResponsePlaybook",
    "ResponseExecutor",
]
