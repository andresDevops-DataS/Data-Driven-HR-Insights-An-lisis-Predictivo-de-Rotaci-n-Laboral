import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "No suministrado por confidencialidad de los datos"


df_turnover = pd.read_excel(file_path, sheet_name='Turnover')
df_headcount = pd.read_excel(file_path, sheet_name='Headcount')

df_turnover_clean = df_turnover.copy()
df_headcount_clean = df_headcount.copy()

df_turnover_clean.columns = df_turnover_clean.columns.str.strip()
df_headcount_clean.columns = df_headcount_clean.columns.str.strip()

for col in df_turnover_clean.select_dtypes(include='object').columns:
    df_turnover_clean[col] = df_turnover_clean[col].astype(str).str.strip()

for col in df_headcount_clean.select_dtypes(include='object').columns:
    df_headcount_clean[col] = df_headcount_clean[col].astype(str).str.strip()


for col in df_turnover_clean.columns:
    if "fecha" in col.lower():
        df_turnover_clean[col] = pd.to_datetime(df_turnover_clean[col], errors='coerce')

for col in df_headcount_clean.columns:
    if "fecha" in col.lower():
        df_headcount_clean[col] = pd.to_datetime(df_headcount_clean[col], errors='coerce')


df_headcount_long = df_headcount_clean.melt(id_vars='Headcount', 
                                             var_name='Año', 
                                             value_name='Valor')
df_headcount_long['Año'] = pd.to_numeric(df_headcount_long['Año'], errors='coerce')


df_headcount_long = df_headcount_long.sort_values(by='Año')


plt.figure(figsize=(10,6))
sns.lineplot(data=df_headcount_long, x='Año', y='Valor', hue='Headcount', marker='o')


for i in range(df_headcount_long.shape[0]):
    x = df_headcount_long['Año'].iloc[i]
    y = df_headcount_long['Valor'].iloc[i]
    zona = df_headcount_long['Headcount'].iloc[i]
    plt.text(x, y + 15, f"{int(y)}", ha='center', fontsize=8)

plt.title('Tendencia del Headcount por Zona')
plt.xlabel('Año')
plt.ylabel('Número de empleados')
plt.grid(True)
plt.tight_layout()
plt.legend(title='Zona')
plt.show()

print("Columnas en TURNOVER:")
print(df_turnover_clean.columns.tolist())

print("\nPrimeras filas de TURNOVER:")
print(df_turnover_clean.head())




df_turnover_clean['Termination date'] = pd.to_datetime(df_turnover_clean['Termination date'], errors='coerce')


df_turnover_clean['Año'] = df_turnover_clean['Termination date'].dt.year


turnover_by_region_year = df_turnover_clean.groupby(['Region', 'Año']).size().reset_index(name='Salidas')


turnover_by_region_year = turnover_by_region_year.dropna(subset=['Año'])


turnover_by_region_year = turnover_by_region_year.sort_values(by='Año')


plt.figure(figsize=(10,6))
sns.lineplot(data=turnover_by_region_year, x='Año', y='Salidas', hue='Region', marker='o')


for i in range(turnover_by_region_year.shape[0]):
    row = turnover_by_region_year.iloc[i]
    plt.text(row['Año'], row['Salidas'] + 0.5, int(row['Salidas']), 
             ha='center', va='bottom', fontsize=9)

plt.title('Salidas de empleados por Zona y Año')
plt.xlabel('Año')
plt.ylabel('Número de salidas')
plt.grid(True)
plt.tight_layout()
plt.legend(title='Zona')
plt.show()




df_headcount_long.rename(columns={'Headcount': 'Region', 'Valor': 'Headcount'}, inplace=True)


df_rotacion = pd.merge(turnover_by_region_year, df_headcount_long, on=['Region', 'Año'], how='inner')


df_rotacion['Tasa de rotación (%)'] = (df_rotacion['Salidas'] / df_rotacion['Headcount']) * 100


df_rotacion['Tasa de rotación (%)'] = df_rotacion['Tasa de rotación (%)'].round(2)


plt.figure(figsize=(10,6))
sns.lineplot(data=df_rotacion, x='Año', y='Tasa de rotación (%)', hue='Region', marker='o')


for i in range(df_rotacion.shape[0]):
    row = df_rotacion.iloc[i]
    plt.text(row['Año'], row['Tasa de rotación (%)'] + 0.5, f"{row['Tasa de rotación (%)']}%", 
             ha='center', va='bottom', fontsize=8)

plt.title('Tasa de Rotación por Zona y Año')
plt.xlabel('Año')
plt.ylabel('Tasa de rotación (%)')
plt.grid(True)
plt.tight_layout()
plt.legend(title='Zona')
plt.show()


tabla_resumen = df_rotacion[['Año', 'Region', 'Headcount', 'Salidas', 'Tasa de rotación (%)']].sort_values(by=['Año', 'Region'])


print("\nTabla resumen de rotación por Zona y Año:")
print(tabla_resumen.head())


output_path = "resumen_rotacion_zona_anio.xlsx"
df_rotacion[['Año', 'Region', 'Headcount', 'Salidas', 'Tasa de rotación (%)']].to_excel(output_path, index=False)

print(f"\n✅ Tabla resumen exportada correctamente a: {output_path}")




df_turnover_clean['Hire date'] = pd.to_datetime(df_turnover_clean['Hire date'], errors='coerce')
df_turnover_clean['Antigüedad (años)'] = (
    (df_turnover_clean['Termination date'] - df_turnover_clean['Hire date']).dt.days / 365
).round(1)


plt.figure(figsize=(10, 6))
sns.histplot(df_turnover_clean['Antigüedad (años)'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Distribución de Antigüedad al Momento de Salida')
plt.xlabel('Antigüedad (años)')
plt.ylabel('Número de empleados')
plt.grid(True)
plt.tight_layout()
plt.savefig("EDA1_antiguedad_salida.png")
plt.show()


plt.figure(figsize=(10, 6))
termination_counts = df_turnover_clean['Reason for termination'].value_counts()
sns.barplot(x=termination_counts.values, y=termination_counts.index, palette='viridis')
plt.title('Motivos de Terminación del Contrato')
plt.xlabel('Número de empleados')
plt.ylabel('Motivo')
plt.grid(True)
plt.tight_layout()
plt.savefig("EDA2_motivos_salida.png")
plt.show()


contract_counts = df_turnover_clean['Type of Contract'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(contract_counts.values, labels=contract_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Distribución por Tipo de Contrato')
plt.tight_layout()
plt.savefig("EDA3_tipo_contrato.png")
plt.show()


gender_counts = df_turnover_clean['Gender'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Distribución por Género de Empleados que Salieron')
plt.tight_layout()
plt.savefig("EDA4_genero.png")
plt.show()


performance_counts = df_turnover_clean['Performance Appraisal'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_counts.index, y=performance_counts.values, palette='coolwarm')
plt.title('Evaluación de Desempeño de Empleados que Salieron')
plt.xlabel('Evaluación')
plt.ylabel('Número de empleados')
plt.grid(True)
plt.tight_layout()
plt.savefig("EDA5_desempeno.png")
plt.show()


eda_df = df_turnover_clean[['Region', 'Gender', 'Type of Contract', 'Antigüedad (años)']].copy()


eda_df = eda_df.dropna(subset=['Antigüedad (años)'])


resumen_region = eda_df.groupby('Region')['Antigüedad (años)'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
resumen_region.rename(columns={
    'count': 'Cantidad',
    'mean': 'Media',
    'median': 'Mediana',
    'std': 'Desviación Estándar',
    'min': 'Mínimo',
    'max': 'Máximo'
}, inplace=True)


resumen_genero = eda_df.groupby('Gender')['Antigüedad (años)'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)


resumen_contrato = eda_df.groupby('Type of Contract')['Antigüedad (años)'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)


with pd.ExcelWriter('resumen_estadistico_eda.xlsx', engine='openpyxl') as writer:
    resumen_region.to_excel(writer, sheet_name='Por Región')
    resumen_genero.to_excel(writer, sheet_name='Por Género')
    resumen_contrato.to_excel(writer, sheet_name='Por Contrato')

print("✅ Tablas estadísticas exportadas a 'resumen_estadistico_eda.xlsx'")




df_numericas = df_turnover_clean.select_dtypes(include=['int64', 'float64'])


if df_numericas.shape[1] >= 2:

    matriz_corr = df_numericas.corr().round(2)


    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='white')
    plt.title('Mapa de Calor de Correlaciones entre Variables Numéricas')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No hay suficientes variables numéricas para generar un heatmap.")

# --- INGENIERÍA DE CARACTERÍSTICAS PARA MODELADO ---


df_modelo = df_turnover_clean.copy()


df_modelo['Hire date'] = pd.to_datetime(df_modelo['Hire date'], errors='coerce')
df_modelo['Termination date'] = pd.to_datetime(df_modelo['Termination date'], errors='coerce')


df_modelo['Antigüedad (años)'] = (df_modelo['Termination date'] - df_modelo['Hire date']).dt.days // 365


df_modelo['Rotación'] = df_modelo['Termination date'].notna().astype(int)


df_modelo['Performance Appraisal'] = df_modelo['Performance Appraisal'].str.extract(r'(\d)').astype(float)


variables_modelo = ['Region', 'Age', 'Type of Contract', 'Performance Appraisal', 'Antigüedad (años)', 'Rotación']
df_modelo = df_modelo[variables_modelo].dropna()


df_modelo_encoded = pd.get_dummies(df_modelo, columns=['Region', 'Type of Contract'], drop_first=True)


print("🔄 DataFrame final para entrenamiento del modelo:")
print(df_modelo_encoded.head())

print(df_modelo['Rotación'].value_counts(normalize=True))

#### IMPLEMENTACION DE ANALISIS DE SUPERVIVENCIA ####

from lifelines import KaplanMeierFitter


df_turnover_clean['duracion_dias'] = (
    df_turnover_clean['Termination date'] - df_turnover_clean['Hire date']
).dt.days


df_turnover_clean = df_turnover_clean.dropna(subset=['duracion_dias'])


df_turnover_clean['evento'] = 1


kmf = KaplanMeierFitter()


kmf.fit(durations=df_turnover_clean['duracion_dias'], event_observed=df_turnover_clean['evento'])


plt.figure(figsize=(10, 6))
kmf.plot(ci_show=True)
plt.title('Curva de Supervivencia (Kaplan-Meier) - Todos los Empleados')
plt.xlabel('Días en la empresa')
plt.ylabel('Probabilidad de permanecer en la empresa')
plt.grid(True)
plt.tight_layout()
plt.savefig("supervivencia_km_general.png")
plt.show()

## COMPARACION POR ZONAS ##

kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

for region in df_turnover_clean['Region'].unique():
    mask = df_turnover_clean['Region'] == region
    kmf.fit(df_turnover_clean[mask]['duracion_dias'], event_observed=df_turnover_clean[mask]['evento'], label=region)
    kmf.plot(ci_show=False)

plt.title('Curvas de Supervivencia por Región')
plt.xlabel('Días en la empresa')
plt.ylabel('Probabilidad de permanecer en la empresa')
plt.grid(True)
plt.legend(title='Región')
plt.tight_layout()
plt.savefig("supervivencia_km_por_region.png")
plt.show()

## DIFERENCIAS ENTRE ZONAS ## 

from itertools import combinations
from lifelines.statistics import logrank_test  # <-- Importar aquí

regiones = df_turnover_clean['Region'].unique()

for r1, r2 in combinations(regiones, 2):
    g1 = df_turnover_clean[df_turnover_clean['Region'] == r1]
    g2 = df_turnover_clean[df_turnover_clean['Region'] == r2]

    result = logrank_test(
        g1['duracion_dias'], g2['duracion_dias'],
        event_observed_A=g1['evento'],
        event_observed_B=g2['evento']
    )
    print(f"{r1} vs {r2} → p-valor: {result.p_value:.4f}")