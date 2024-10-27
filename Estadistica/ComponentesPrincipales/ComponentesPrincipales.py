import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Estadistica/ComponentesPrincipales/Country-data.csv")

X = data.drop(columns=['country', 'gdpp'])  
y = data['gdpp']           


scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)
X_Scaled = pd.DataFrame(X_Scaled, columns=X.columns)

# Regresión Lineal
print("\n\n#################################")
print("                                 ")
print("        Regresión Lineal:        ")
print("                                 ")
print("#################################\n\n")

X_scaled_const = sm.add_constant(X_Scaled)
regresion = sm.OLS(y, X_scaled_const).fit()
print(regresion.summary())

#################################
# Con un R^2 de 0.866 el modelo de regresion lineal multiple se ajusta bien a los datos
# 
# "Health", "income" y "life_expec" son significativas con p-valores menores a 0.05.
# Algunas variables, como "exports", "imports" y "total_fer", no son significativas, 
#
# Al tener un valor de Durbin-Watson de 1.9 (cercano a 2) podemos decir que hay poca 
# autocorrelación entre los residuos
#################################


# Matriz de covarianza
print("\n\n#################################")
print("                                 ")
print("      Matriz de covarianza:      ")
print("                                 ")
print("#################################\n\n")
#################################
# Hay correlaciones importantes entre variables socioeconómicas y de salud. 
# La esperanza de vida parece estar fuertemente relacionada con:
#   -  child_mort: -0.89
#   -  total_fer:   0.76
#   -  income:      0.61
# Parece haber una correlación significativa entre total_fer y child_mort con 0.85
# Parece haber una correlación significativa entre imports y exports con 0.74
#################################

cov_matrix = np.cov(X_Scaled, rowvar=False)
cov_df = pd.DataFrame(cov_matrix, index=X.columns, columns=X.columns)
print(cov_df)


print("\n\n#################################")
print("                                 ")
print("  Valores y vectores propios:    ")
print("                                 ")
print("#################################\n\n")
################################# 
# Al igual que en la matriz de covarianza y según el analisis 
# de regresión lineal multiple, podemos ver que en ciertos casos
# como el del vector propio 1, se ve mas influenciado por variables
# como child_mort, life_expec, y total_fer.
#################################

valorespropios, vectorespropios = np.linalg.eig(cov_matrix)
valorespropios_df = pd.DataFrame(valorespropios, index=[f'Valor Propio {i+1}' for i in range(len(valorespropios))], columns=['Valor propio'])
vectorespropios_df = pd.DataFrame(vectorespropios, columns=[f'Vector Propio {i+1}' for i in range(len(vectorespropios))], index=X.columns)

print("Valores propios:")
print(valorespropios_df)
print("\nVectores propios:")
print(vectorespropios_df)


# Varianza y n_componentes
varianza_explicada = valorespropios / np.sum(valorespropios)
varianza_acumulada = np.cumsum(varianza_explicada)
n_componentes = np.argmax(varianza_acumulada >= 0.80) + 1

print(f"\nNúmero de componentes principales seleccionados: {n_componentes}")
print(f"\nVarianza acumulada por los primeros {n_componentes} componentes: {varianza_acumulada[n_componentes-1]:.2f}")

print("\n\n#################################")
print("                                 ")
print("     Componentes principales y   ")
print("    ecuaciones de transformación ")
print("                                 ")
print("#################################\n\n")
#################################
# El componente principal 1 se ve mas influenciado por:
# - child_mort, life_expec, total_fert e income
#
# El componente principal 2 se ve mas influenciado por:
# - exports, imports y health
#
# El componente principal 3 se ve mas influenciado por:
# - inflation, health  e imports
#
# El componente principal 4 se ve mas influenciado por:
# - inflation, health e income
#################################

componentes_variables = pd.DataFrame(vectorespropios_df)
componentes_variables.columns = [f'Componente {i+1}' for i in range(len(componentes_variables.columns))]
contribucion_significativa = pd.DataFrame()
ecuaciones_componentes = {}
num_variables_significativas = 3  # Variables más influyentes en cada componente

for i in range(n_componentes):
    # Ordenar las variables según su contribución en valor absoluto para el componente actual
    contribucion = componentes_variables.iloc[:, i].abs().sort_values(ascending=False)
    contribucion_significativa[f'Componente Principal {i+1}'] = contribucion
    
    # Seleccionar las variables significativas y formar la ecuación lineal
    variables_significativas = contribucion.head(num_variables_significativas)
    ecuacion = " + ".join(
        [f"{vectorespropios_df.loc[var, f'Vector Propio {i+1}']:.2f} * {var}" for var in variables_significativas.index]
    )
    ecuaciones_componentes[f'Componente Principal {i+1}'] = ecuacion

# Mostrar contribución de variables a cada componente principal
print("\nContribución de Variables a Cada Componente Principal:")
print(contribucion_significativa)

# Mostrar ecuaciones de transformación para cada componente
print("\nEcuaciones de Transformación de Componentes Principales:")
for componente, ecuacion in ecuaciones_componentes.items():
    print(f"{componente}: {ecuacion}")


# Crear un DataFrame con los datos transformados
print("\n\n#################################")
print("                                 ")
print("       Datos Transformados:      ")
print("                                 ")
print("#################################\n\n")

# Transformar los datos al nuevo espacio con los componentes seleccionados
vectores_seleccionados = vectorespropios[:, :n_componentes]
X_transformado = np.dot(X_Scaled, vectores_seleccionados)
X_transformado_df = pd.DataFrame(X_transformado, columns=[f'Componente Principal{i+1}' for i in range(n_componentes)])
print("\nDatos transformados al nuevo espacio de componentes principales:")
print(X_transformado_df)


print("\n\n#################################")
print("                                 ")
print("    Nuevo Modelo de Regresión:   ")
print("                                 ")
print("#################################\n\n")
#################################
# Queríamos que el nuevo modelo de regresión cumpliera con el 80% de la varianza acumalada, y lo logra con los primeros 4 componentes
# teniendo una varianza acumulada del 0.88
#
# Desafortunadamente nuestro R^2 bajó de 0.86 a 0.610
# Tomando esto en cuenta podemos ver que los Componentes 1 y 4 son altamente significativos con un valor P muy cercano a 0.
##################################

X_transformado_const = sm.add_constant(X_transformado_df.iloc[:, :n_componentes])
regresion_principales = sm.OLS(y, X_transformado_const).fit()
print(regresion_principales.summary())


# Los clusters es lo unico que si no terminé de entender
kmeans = KMeans(n_clusters=3, random_state=42) 
clusters = kmeans.fit_predict(X_transformado_df)
X_transformado_df['Cluster'] = clusters

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    X_transformado_df['Componente Principal1'], 
    X_transformado_df['Componente Principal2'], 
    X_transformado_df['Componente Principal3'], 
    c=X_transformado_df['Cluster'], 
    cmap='viridis', 
    s=60
)

ax.set_title("Visualización de Clusters en 3D utilizando Componentes Principales")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.set_zlabel("Componente Principal 3")
plt.colorbar(sc, label='Cluster')
plt.show()
