import pandas as pd

# Lê a planilha CSV
df = pd.read_csv('alzheimers_disease_data.csv')

# Exclui a primeira coluna (ID) e a última coluna (Nome do Médico)
df_filtered = df.iloc[:, 1:-1]

# Calcula as estatísticas
estatisticas = df_filtered.describe().T

# Adiciona a mediana
estatisticas['median'] = df_filtered.median()

# Reorganiza as colunas para incluir a mediana entre as outras estatísticas
estatisticas = estatisticas[['mean', 'std', '50%', 'min', 'max']]
estatisticas.columns = ['mean', 'std', 'median', 'min', 'max']

# Exibe as estatísticas
print(estatisticas)

# Salva as estatísticas em um novo arquivo CSV
estatisticas.to_csv('estatisticas.csv')
