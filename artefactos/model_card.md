# Model Card — MLP (Regresión)
**Versión:** 20251119_002224  
**Entorno:** Python 3.12.7 | scikit-learn 1.5.1

## Datos
Archivo: `Wellbeing_and_lifestyle_data_Kaggle.csv`  
Shape: (15972, 22)  
Objetivo: `WORK_LIFE_BALANCE_SCORE`

## Entrenamiento
Split 80/20 (random_state=42)
Preprocesamiento: StandardScaler(num)

## Métricas TEST
MAE=0.282 | RMSE=0.415 | R²=1.000

## Artefactos
- pipeline_MLP.joblib
- input_schema.json
- decision_policy.json
