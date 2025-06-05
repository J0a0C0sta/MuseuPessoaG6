#Normalização do texto
entrevistaosl.txt: 9_Se_pudesse_criar_uma_nova_disciplina.txt
	onesentpl -x 9_Se_pudesse_criar_uma_nova_disciplina.txt > entrevistaosl.txt

# Frequência das palavras e rácio
entrevista-freq.txt: entrevistaosl.txt
	python "C:\Users\marco\OneDrive\Documentos\AVD2425\Tools\Freq/freq.py" -r entrevistaosl.txt > entrevista-freq.txt


# Lemas e lemas sem pontuação
entrevistacon.txt: entrevistaosl.txt
	avdconll entrevistaosl.txt > entrevistacon.txt
	rg -v PUNCT entrevistacon.txt > entrevistaspunct.txt

#Extração de palavras-chave
palavra-chave.txt: entrevistaosl.txt
	python "C:/Users/marco/OneDrive/Documentos/AVD2425/Tools/T1/keyword_extract.py" -n 5 entrevistaosl.txt > palavra-chave.txt


# ANÁLISE DE SENTIMENTO:

# Análise sentimento
entrevista-sentimento.txt: entrevistaosl.txt
	sentimento entrevistaosl.txt > entrevista-sentimento.txt

# Análise sentimento detalhada
entrevista-sentimento-detail.txt: entrevistaosl.txt
	sentimento -o entrevistaosl.txt > entrevista-sentimento-detail.txt

# ANÁLISE DETALHADA COM RÁCIO DE PALAVRAS:

# Análise detalhada 
entrevista-ratio.txt: entrevistaosl.txt
	sentimento_detalhado entrevistaosl.txt > entrevista-ratio.txt

# Análise detalhada em formato tabela (TSV) 
entrevista-summary.txt: entrevista.txt
	sentimento_detalhado -s entrevistaosl.txt > entrevista-summary.txt