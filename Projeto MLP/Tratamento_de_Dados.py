import pandas as pd
file_path = 'alzheimers_disease_data.csv'

# Leia o arquivo CSV
df = pd.read_csv(file_path)

# Verifique se o arquivo tem 35 colunas
if df.shape[1] != 35:
    print(f"O arquivo deve ter 35 colunas, mas tem {df.shape[1]}")
else:
    print("O arquivo tem 35 colunas.")

# Verifique se há valores faltando
missing_values = df.iloc[:, :-1].isnull().sum().sum()  # Ignora a última coluna

if missing_values > 0:
    print(f"Existem {
          missing_values} valores faltando nas colunas.")
else:
    print("Não há valores faltando nas colunas.")

# Verifique se há strings nas colunas, excluindo a última coluna "Informação do médico"
string_present = False
for col in df.columns[:-1]:  # Exclui a última coluna do doutor
    if df[col].apply(lambda x: isinstance(x, str)).any():
        print(f"A coluna '{col}' contém strings.")
        string_present = True

if not string_present:
    print("Não há strings em nenhuma coluna, excluindo a última coluna.")
