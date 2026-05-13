# agents.md — Proyecto Final de Inteligencia Artificial
## Predicción de Ventas en Tienda de Café (Coffee Shop Sales)

---

## Contexto del Proyecto

**Dataset:** `Coffee_Shop_Sales.csv`
**Registros:** ~149,116 transacciones
**Objetivo:** Predecir el total de ingresos por transacción (regresión) usando modelos Ensemble optimizados

**Variables disponibles:**
| Variable | Tipo | Descripción |
|---|---|---|
| `transaction_id` | int | Identificador único |
| `transaction_date` | str | Fecha de la transacción |
| `transaction_time` | str | Hora de la transacción |
| `transaction_qty` | int | Cantidad de productos vendidos |
| `store_id` | int | ID de la tienda |
| `store_location` | str | Ubicación de la tienda |
| `product_id` | int | ID del producto |
| `unit_price` | float | Precio unitario |
| `product_category` | str | Categoría del producto |
| `product_type` | str | Tipo de producto |
| `product_detail` | str | Detalle del producto |

**Variable objetivo a crear:** `total_revenue = transaction_qty × unit_price`

---

## Roadmap de Clases (Progresión del Proyecto)

```
Clase 5: Validación ML          Clase 6: Árbol Decisión       Clase 7: Ensemble + Optimización
        |                               |                               |
        v                               v                               v
  Train/Test Split    ------->   Árbol de Decisión   ------->   Random Forest
  Cross-Validation               (baseline del proyecto)         XGBoost
  EDA + Limpieza                 Feature Engineering             LightGBM
                                                                 Grid Search / Optuna
```

---

## FASE 0 — Preparación del Entorno

```python
# Librerías necesarias (instalar si no están)
# pip install lightgbm xgboost optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import optuna
```

---

## FASE 1 — Carga y Exploración de Datos (EDA)

**Referencia:** Clase 5 — Validación ML

```python
# 1.1 Cargar el dataset
df = pd.read_csv('Coffee_Shop_Sales.csv')

# 1.2 Exploración básica
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
print(df.head())
```

### Preguntas clave a responder durante el EDA:
- ¿Cuáles son las categorías de productos más vendidas?
- ¿Hay diferencias de ventas por tienda (`store_location`)?
- ¿Existen patrones de ventas por hora o día de la semana?
- ¿Hay valores nulos o outliers en `unit_price` o `transaction_qty`?

```python
# 1.3 Análisis de distribuciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df['total_revenue'].hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Distribución de Ingresos por Transacción')

df.groupby('product_category')['total_revenue'].sum().sort_values().plot(
    kind='barh', ax=axes[0, 1])
axes[0, 1].set_title('Ingresos Totales por Categoría')

df.groupby('store_location')['total_revenue'].mean().plot(
    kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Ingreso Promedio por Tienda')

df.groupby('transaction_qty')['total_revenue'].mean().plot(
    kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Ingreso Promedio por Cantidad')

plt.tight_layout()
plt.savefig('eda_overview.png')
plt.show()
```

---

## FASE 2 — Feature Engineering (Ingeniería de Variables)

**Referencia:** Clase 6 — Árbol de Decisión (variables que alimentan los nodos de decisión)

```python
# 2.1 Crear variable objetivo
df['total_revenue'] = df['transaction_qty'] * df['unit_price']

# 2.2 Extraer features de fecha y hora
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')

df['day_of_week'] = df['transaction_date'].dt.dayofweek     # 0=Lunes, 6=Domingo
df['month'] = df['transaction_date'].dt.month
df['day_of_month'] = df['transaction_date'].dt.day
df['hour'] = df['transaction_time'].dt.hour
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)

# 2.3 Codificar variables categóricas
le = LabelEncoder()
cat_cols = ['store_location', 'product_category', 'product_type', 'product_detail']
for col in cat_cols:
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

# 2.4 Definir features y target
feature_cols = [
    'transaction_qty', 'store_id', 'product_id', 'unit_price',
    'store_location_encoded', 'product_category_encoded',
    'product_type_encoded', 'product_detail_encoded',
    'day_of_week', 'month', 'day_of_month', 'hour',
    'is_weekend', 'is_morning', 'is_afternoon'
]

X = df[feature_cols]
y = df['total_revenue']

print(f"Features: {X.shape[1]} columnas")
print(f"Registros: {X.shape[0]}")
print(f"Target - Rango: [{y.min():.2f}, {y.max():.2f}] | Media: {y.mean():.2f}")
```

---

## FASE 3 — División y Validación de Datos

**Referencia:** Clase 5 — Train/Test Split y K-Fold Cross-Validation

```python
# 3.1 División Train/Test (80/20) — Clase 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 3.2 Configurar K-Fold para Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Función utilitaria de métricas
def evaluar_modelo(y_true, y_pred, nombre="Modelo"):
    r2  = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {nombre}")
    print(f"{'='*50}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}
```

---

## FASE 4 — Entrenamiento de Modelos Ensemble

**Referencia:** Clase 7 — Algoritmos Ensemble y Optimización de Hiperparámetros

> **Tarea obligatoria:** Entrenar Random Forest, XGBoost y LightGBM. Comparar R², RMSE y MAE.

### 4.1 Random Forest (Bagging)
```python
inicio = time.time()

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)

# Cross-Validation primero
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
print(f"RF - R² CV: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")

# Entrenar en todo el train set y evaluar en test
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
tiempo_rf = time.time() - inicio

resultados_rf = evaluar_modelo(y_test, rf_pred, "Random Forest")
print(f"  Tiempo : {tiempo_rf:.1f}s")
```

### 4.2 XGBoost (Boosting)
```python
inicio = time.time()

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='r2')
print(f"XGB - R² CV: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std()*2:.4f})")

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
tiempo_xgb = time.time() - inicio

resultados_xgb = evaluar_modelo(y_test, xgb_pred, "XGBoost")
print(f"  Tiempo : {tiempo_xgb:.1f}s")
```

### 4.3 LightGBM (Boosting — el más rápido)
```python
inicio = time.time()

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

lgb_cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=kf, scoring='r2')
print(f"LGB - R² CV: {lgb_cv_scores.mean():.4f} (+/- {lgb_cv_scores.std()*2:.4f})")

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
tiempo_lgb = time.time() - inicio

resultados_lgb = evaluar_modelo(y_test, lgb_pred, "LightGBM")
print(f"  Tiempo : {tiempo_lgb:.1f}s")
```

### 4.4 Tabla Comparativa de Modelos
```python
tabla = pd.DataFrame({
    'Modelo':  ['Random Forest', 'XGBoost', 'LightGBM'],
    'R²':      [resultados_rf['R2'],   resultados_xgb['R2'],   resultados_lgb['R2']],
    'RMSE':    [resultados_rf['RMSE'], resultados_xgb['RMSE'], resultados_lgb['RMSE']],
    'MAE':     [resultados_rf['MAE'],  resultados_xgb['MAE'],  resultados_lgb['MAE']],
    'Tiempo_s': [tiempo_rf, tiempo_xgb, tiempo_lgb]
})
print(tabla.to_string(index=False))
mejor_modelo_nombre = tabla.loc[tabla['R²'].idxmax(), 'Modelo']
print(f"\n✅ Mejor modelo: {mejor_modelo_nombre}")
```

---

## FASE 5 — Optimización de Hiperparámetros

**Referencia:** Clase 7 — Grid Search y Optuna

> **Tarea obligatoria:** Optimizar el mejor modelo. Documentar hiperparámetros probados y mejora obtenida.

### Opción A: Grid Search (sklearn)
```python
# Aplicar al mejor modelo (ejemplo con LightGBM)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [6, 10, 15],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves':   [20, 31, 50]
}

grid_search = GridSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor R² (CV): {grid_search.best_score_:.4f}")
```

### Opción B: Optuna (búsqueda bayesiana — más eficiente)
```python
def objetivo_optuna(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'max_depth':     trial.suggest_int('max_depth', 4, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves':    trial.suggest_int('num_leaves', 20, 100),
        'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objetivo_optuna, n_trials=30, show_progress_bar=True)

print(f"Mejor R² encontrado: {study.best_value:.4f}")
print(f"Mejores parámetros:\n{study.best_params}")
```

### Evaluar Modelo Optimizado vs Base
```python
modelo_opt = lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
modelo_opt.fit(X_train, y_train)
opt_pred = modelo_opt.predict(X_test)
resultados_opt = evaluar_modelo(y_test, opt_pred, "LightGBM Optimizado")

print(f"\n📊 Mejora obtenida:")
print(f"  R² base      : {resultados_lgb['R2']:.4f}")
print(f"  R² optimizado: {resultados_opt['R2']:.4f}")
print(f"  Mejora       : +{(resultados_opt['R2'] - resultados_lgb['R2'])*100:.2f}%")
```

---

## FASE 6 — Feature Importance

**Referencia:** Clase 7 — Feature Importance + Clase 6 — Árbol de Decisión (qué variables usa el árbol para dividir)

> **Tarea obligatoria:** Gráfico de feature importance + identificar las 5 variables más importantes y explicarlas.

```python
# 6.1 Obtener y graficar importancias
importancias = pd.Series(
    modelo_opt.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importancias.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — LightGBM Optimizado', fontsize=14)
plt.xlabel('Importancia Relativa')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# 6.2 Top 5 variables
top5 = importancias.sort_values(ascending=False).head(5)
print("\n🏆 Top 5 Variables Más Importantes:")
for i, (feat, imp) in enumerate(top5.items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")
```

**Plantilla de análisis (completar con los resultados obtenidos):**
```
Top 5 Variables — Análisis:

1. [variable] — Tiene sentido porque...
2. [variable] — Tiene sentido porque...
3. [variable] — Tiene sentido porque...
4. [variable] — Tiene sentido porque...
5. [variable] — Tiene sentido porque...
```

---

## FASE 7 — Análisis de Tiempo (Opcional)

**Referencia:** Clase 7 — Lección "Más No Siempre Es Mejor"

```python
# Comparar tiempos y rendimiento de todos los modelos
resumen_tiempo = pd.DataFrame({
    'Modelo':   ['Random Forest', 'XGBoost', 'LightGBM', 'LightGBM Optimizado'],
    'R²':       [resultados_rf['R2'], resultados_xgb['R2'],
                 resultados_lgb['R2'], resultados_opt['R2']],
    'Tiempo_s': [tiempo_rf, tiempo_xgb, tiempo_lgb, '—']
})

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(resumen_tiempo))
ax.bar(x, resumen_tiempo['R²'], color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'])
ax.set_xticks(x)
ax.set_xticklabels(resumen_tiempo['Modelo'], rotation=15)
ax.set_ylabel('R²')
ax.set_title('Comparación de Modelos — R² en Test Set')
plt.tight_layout()
plt.savefig('comparacion_modelos.png', dpi=150)
plt.show()
```

---

## ENTREGABLE FINAL

El notebook debe contener, en orden:

| # | Sección | Obligatorio |
|---|---|---|
| 1 | EDA + gráficos de exploración | ✅ |
| 2 | Feature Engineering (creación de variables) | ✅ |
| 3 | Train/Test Split + configuración CV | ✅ |
| 4 | Entrenamiento RF, XGBoost, LightGBM | ✅ |
| 5 | Tabla comparativa (R², RMSE, MAE) | ✅ |
| 6 | Optimización (Grid Search u Optuna) | ✅ |
| 7 | Gráfico de Feature Importance | ✅ |
| 8 | Análisis de las 5 variables + explicación | ✅ |
| 9 | Análisis de tiempos de entrenamiento | ⬜ Opcional |
| 10 | Conclusiones en texto | ✅ |

### Plantilla de Conclusiones
```
## Conclusiones del Proyecto

### 1. Mejor Modelo
El modelo con mejor desempeño fue [MODELO] con un R² de [VALOR],
lo que significa que explica el [X]% de la varianza en los ingresos.

### 2. Variables Más Importantes
Las variables que más influyen en las ventas son [LISTAR],
lo que tiene sentido porque [EXPLICACIÓN BREVE].

### 3. Optimización
La optimización con [Grid Search / Optuna] mejoró el R² en [X]%.
Los hiperparámetros más relevantes fueron [PARÁMETROS].

### 4. Aprendizajes
- Aprendizaje 1...
- Aprendizaje 2...
- Aprendizaje 3...
```

---

## Criterios de Evaluación

| Criterio | Descripción |
|---|---|
| **Código ejecutable** | El notebook corre de principio a fin sin errores |
| **Tres modelos** | RF, XGBoost y LightGBM entrenados y comparados |
| **Optimización** | Se documenta qué se probó y cuánto mejoró |
| **Feature Importance** | Gráfico + explicación escrita de las 5 top |
| **Conclusiones** | Reflexión clara sobre los resultados |

---

*Proyecto basado en: Clase 5 (Validación ML), Clase 6 (Árbol de Decisión), Clase 7 (Ensemble y Optimización)*
*Dataset: Coffee_Shop_Sales.csv — ~149,116 transacciones reales de una cadena de cafeterías*
