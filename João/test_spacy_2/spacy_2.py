import spacy
import sys
import pandas as pd

# Carrega o modelo spaCy
pln = spacy.load("pt_core_news_lg")

# Lê o arquivo passado como argumento
file = sys.argv[1]
with open(file, encoding="utf-8") as f:
    txt = f.read()

# Processa o texto
ad = pln(txt)

# Cria listas para armazenar lemas e entidades
lemmas = []
entities = []

# Adiciona lemas, ignorando pontuação e espaços
for token in ad:
    if not token.is_punct and not token.is_space:
        lemmas.append((token.text, token.lemma_))

# Adiciona entidades
for ent in ad.ents:
    entities.append((ent.text, ent.label_, spacy.explain(ent.label_)))

# Cria DataFrames para os lemas e entidades
df_lemmas = pd.DataFrame(lemmas, columns=["Palavra", "Lema"])
df_entities = pd.DataFrame(entities, columns=["Entidade", "Tipo", "Explicação"])

# Caminho para salvar o arquivo Excel (Usando raw string)
output_path = r"C:\Users\joaop\Desktop\teste_spacy\resultado_2.xlsx"

# Escreve os DataFrames em abas separadas no Excel
with pd.ExcelWriter(output_path) as writer:
    df_lemmas.to_excel(writer, sheet_name="Lemas", index=False)
    df_entities.to_excel(writer, sheet_name="Entidades", index=False)

print(f"Arquivo exportado com sucesso para: {output_path}")