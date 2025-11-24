# Model Card — RG (Regresión)
**Versión:** 20251124_204516

## Datos
Archivo: `Wellbeing_and_lifestyle_data_Kaggle.csv`
Shape: (15972, 22)
Objetivo: `WORK_LIFE_BALANCE_SCORE`

## Entrenamiento
- Split 80/20
- Preprocesamiento: StandardScaler(num) + OneHotEncoder(cat)
- Modelo final: RG

## Métricas TEST
- MAE = 0.0000
- RMSE = 0.0000
- R² = 1.0000

## Artefactos
- pipeline_RG.joblib
- input_schema.json
- decision_policy.json
