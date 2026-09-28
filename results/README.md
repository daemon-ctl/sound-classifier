# Experiment results

Таблицы экспортированы из финального сравнения CNN Width-8 и CRNN Micro:

- `metrics_by_seed.csv` и `metrics_summary.csv` — общие метрики;
- `per_class_metrics_summary.csv` — показатели по классам;
- `latency.csv` и `complexity.csv` — задержка и размер моделей;
- `errors.csv` — ошибки классификации для анализа.
- `baseline_comparison.csv`, `data_saturation.csv`, `augmentation_*.csv` и
  `model_size_ablation.csv` — ключевые промежуточные эксперименты, из которых
  строится исследовательская цепочка финального ноутбука.
- `augmentation_human_validation.csv` и `augmentation_robustness_diagnostic.csv` —
  исходные ручные оценки и модельная диагностика аугментаций.

Исходные предсказания и истории обучения намеренно не включены в корень репозитория;
при необходимости их можно добавить как расширенные артефакты эксперимента.
