# Data-Driven HR Insights: Análisis Predictivo de Rotación Laboral

Este proyecto surge como respuesta a una **prueba técnica** realizada en el marco de un proceso de selección para una empresa del sector industrial. Por curiosidad y entusiasmo, decidí abordar el reto como un proyecto completo de análisis de datos y ciencia aplicada al talento humano.

## 📌 Objetivos del proyecto

- Analizar la rotación de personal histórica.
- Estimar tasas de rotación por zona y por año.
- Visualizar tendencias en salidas y contrataciones.
- Construir un modelo predictivo de rotación basado en datos disponibles.
- Aplicar análisis de supervivencia para comprender la duración laboral.

## 📊 Contenido del análisis

- Análisis exploratorio (EDA)
- Gráficos por región, género, contrato, etc.
- Cálculo de métricas de rotación
- Árbol de decisión como modelo de clasificación
- Curvas de supervivencia (Kaplan-Meier)
- Pruebas de hipótesis entre zonas (log-rank test)

## 🛠️ Herramientas y librerías usadas

- Python (pandas, matplotlib, seaborn, scikit-learn)
- lifelines (para análisis de supervivencia)
- Jupyter Notebook
- Visual Studio Code

## 📁 Estructura del repositorio
├── EDA_SW.py # Script principal de análisis
├── figuras/ # Visualizaciones generadas
├── modelo_predictivo.pkl # Modelo de Analisis de supervivencia
├── README.md # Descripción del proyecto

## 📈 Resultados clave

- Identificación de las zonas con mayores tasas de rotación.
- Árbol de decisión con precisión superior al 90%.
- Comprobación estadística de que no hay diferencias significativas en la duración laboral entre zonas pero hay existencia de correlacion entre antiguedad y mayor probabilidad de salida.

## 📣 Autor

**Andres Felipe Jimenez Hernandez**  
GitHub: [@andresDevops-DataS](https://github.com/andresDevops-DataS)

---

Este proyecto refleja cómo la ciencia de datos puede generar valor en áreas clave como Recursos Humanos. 🚀
