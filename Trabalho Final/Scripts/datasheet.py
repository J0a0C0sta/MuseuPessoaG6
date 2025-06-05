import re
import csv

input_file = '10_analisada.txt'
output_file = '10_analise_convertida.csv'

# Regex patterns para apanhar os dados
paragrafo_pat = re.compile(r'PARÁGRAFO (\d+): (\w+)\nPreview: (.+)')
polaridade_pat = re.compile(r'Polaridade Geral: ([\-\d.]+)\nPolaridade Média das Palavras: ([\-\d.]+)')
estatisticas_pat = re.compile(
    r'Total de palavras analisadas: (\d+)\n  Palavras positivas: (\d+) \(([\d.]+)%\)\n  Palavras negativas: (\d+) \(([\d.]+)%\)\n  Palavras neutras: (\d+) \(([\d.]+)%\)'
)
mais_positivas_pat = re.compile(r'PALAVRAS MAIS POSITIVAS:[\s\S]*?([a-zãõáéíóúç-]+).*?: ([\-\d.]+)')
mais_negativas_pat = re.compile(r'PALAVRAS MAIS NEGATIVAS:[\s\S]*?([a-zãõáéíóúç-]+).*?: ([\-\d.]+)')

with open(input_file, encoding='utf-8') as fin, open(output_file, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    # Escreve cabeçalho
    writer.writerow([
        'Parágrafo', 'Sentimento', 'Preview', 'Polaridade Geral', 'Polaridade Média',
        'Total palavras', 'Positivas', '% Positivas', 'Negativas', '% Negativas', 'Neutras', '% Neutras',
        'Mais Positiva', 'Score Mais Positiva', 'Mais Negativa', 'Score Mais Negativa'
    ])
    text = fin.read()
    blocks = text.split('============================================================')
    for block in blocks:
        block = block.strip()
        if not block: continue

        par_match = paragrafo_pat.search(block)
        pol_match = polaridade_pat.search(block)
        est_match = estatisticas_pat.search(block)
        pos_match = mais_positivas_pat.search(block)
        neg_match = mais_negativas_pat.search(block)

        if par_match and pol_match and est_match:
            row = [
                par_match.group(1),
                par_match.group(2),
                par_match.group(3),
                pol_match.group(1),
                pol_match.group(2),
                est_match.group(1),
                est_match.group(2),
                est_match.group(3),
                est_match.group(4),
                est_match.group(5),
                est_match.group(6),
                est_match.group(7),
                pos_match.group(1) if pos_match else "",
                pos_match.group(2) if pos_match else "",
                neg_match.group(1) if neg_match else "",
                neg_match.group(2) if neg_match else "",
            ]
            writer.writerow(row)
print("CSV criado com sucesso!")
