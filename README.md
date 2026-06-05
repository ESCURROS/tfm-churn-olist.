# Predicción de Churn en E-Commerce — Olist

Aplicación web de predicción de abandono de clientes (*churn*) para el marketplace
brasileño **Olist**, desarrollada como Trabajo Fin de Máster del **Máster en Data Science
Executive (IMF Smart Education)**.

**Autor:** Eduardo Jose Sanz Curros
**Categoría:** Data Science · **Convocatoria:** 2026

🔗 **App en vivo:** https://tfm-churn-olist-6pmhs7kd6bn6qjblzh5eut.streamlit.app/

---

## 1. Descripción del proyecto

El objetivo es predecir, mediante aprendizaje automático supervisado, la probabilidad de
que un cliente de Olist **no realice ninguna compra en los 92 días siguientes** a una fecha
de corte (*cutoff*). El proyecto cubre el pipeline completo de ciencia de datos: adquisición
e integración de las 9 tablas relacionales del dataset, análisis exploratorio (EDA),
ingeniería de variables RFM con **separación temporal estricta** para evitar *data leakage*,
balanceo de clases con SMOTE, comparativa de cuatro algoritmos, interpretabilidad con SHAP y
despliegue de esta aplicación interactiva.

## 2. Dataset

- **Fuente:** Brazilian E-Commerce Public Dataset by Olist (Kaggle)
- **Enlace:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Volumen:** 9 tablas relacionales · 96.478 pedidos entregados (2016–2018)
- **Licencia:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0

## 3. Metodología

1. **Definición temporal de churn.** Período de observación hasta el *cutoff* (01/06/2018)
   para calcular las variables; período de predicción de 92 días (01/06/2018 → 01/09/2018)
   para definir la etiqueta. Solo se incluyen clientes con historial previo al *cutoff*
   (75.387 clientes únicos).
2. **Feature engineering (RFM + complementarias):** `recencia_dias`, `frecuencia`,
   `valor_total`, `avg_review_score`, `num_categorias`, `avg_delivery_days`.
3. **Corrección de data leakage.** La versión inicial calculaba la recencia sobre todo el
   dataset, haciéndola equivalente a la etiqueta (AUC-ROC = 1.0, irrealista). Al aplicar la
   ventana temporal, la correlación recencia↔churn pasó de 0.79 a 0.020, eliminando el leakage.
4. **Balanceo:** SMOTE aplicado **solo sobre el conjunto de entrenamiento**.
5. **Modelado:** comparativa de Logistic Regression, Random Forest, XGBoost y LightGBM.
6. **Evaluación e interpretabilidad:** AUC-ROC, F1, Precision, Recall, validación cruzada
   5-fold y análisis SHAP.

## 4. Resultados

Sobre un conjunto de test con la distribución real (desbalanceo extremo: 99,4 % churn):

| Modelo | AUC-ROC | F1-Score | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.6038** | 0.7197 | 0.9959 | 0.5635 |
| Random Forest | 0.5051 | 0.9613 | 0.9945 | 0.9302 |
| XGBoost | 0.4904 | 0.9948 | 0.9946 | 0.9951 |
| LightGBM | 0.4880 | 0.9949 | 0.9947 | 0.9952 |

El modelo seleccionado para producción es la **Regresión Logística**, único con capacidad
discriminativa por encima del azar. En un escenario con tasa base del 99,4 %, el AUC-ROC es
más informativo que el F1-Score.

## 5. La aplicación

La app (Streamlit) tiene cuatro secciones:

- **Inicio y metodología** — KPIs del dataset, pipeline y nota sobre la corrección del leakage.
- **Predicción individual** — *sliders* para introducir datos de un cliente y obtener su
  probabilidad de churn en tiempo real.
- **Predicción por lotes** — carga de un CSV de clientes y descarga de las predicciones.
- **Rendimiento del modelo** — comparativa de los cuatro modelos y métricas.

## 6. Estructura del repositorio

```
.
├── app.py                       # Aplicación Streamlit
├── modelo_churn.pkl             # Modelo entrenado (Logistic Regression)
├── scaler_churn.pkl             # StandardScaler ajustado en entrenamiento
├── modelo_metadatos.json        # Métricas, features y configuración del modelo
├── olist_sample_streamlit.csv   # Muestra de clientes para la demo
├── requirements.txt             # Dependencias
└── runtime.txt                  # Versión de Python (3.12)
```

## 7. Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

## 8. Stack tecnológico

Python 3.12 · Streamlit · scikit-learn 1.6.1 · pandas · NumPy · joblib · Matplotlib · seaborn

## 9. Limitaciones y trabajo futuro

Olist es un marketplace de compra mayoritariamente única (>90 % de clientes con un solo
pedido), lo que convierte la predicción de recompra a 92 días en un problema de evento raro.
Como trabajo futuro: ventanas de predicción alternativas, calibración de probabilidades,
features de navegación/CRM y modelos de supervivencia.

---

*Trabajo Fin de Máster · Máster en Data Science Executive · IMF Smart Education · 2026*
