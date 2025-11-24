# Model Card — LR (Regresión)
**Versión:** 20251124_190919  
**Entorno:** Python 3.12.7 | scikit-learn 1.5.1

## Datos
Archivo: `Wellbeing_and_lifestyle_data_Kaggle.csv`  
Shape: (15972, 29)  
Objetivo: `WORK_LIFE_BALANCE_SCORE`

## Entrenamiento
Split 80/20 (random_state=42)
Preprocesamiento: StandardScaler(num)

## Métricas TEST
MAE=0.000 | RMSE=0.000 | R²=1.000

## Artefactos
- pipeline_LR.joblib
- input_schema.json
- decision_policy.json
