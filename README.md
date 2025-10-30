# Detecao de Fraudes em Cartoes de Credito

Projeto desenvolvido a partir do conjunto de dados `creditcard - menor balanceado.csv` e das diretrizes do documento `Trab. Machine Learning.pdf` (ambos devem ser adicionados manualmente na raiz do projeto, pois nao sao versionados). O objetivo e comparar diferentes tecnicas de machine learning para identificar transacoes fraudulentas em um cenario desbalanceado.

## Visao geral
- **Pre-processamento**: padronizacao (`StandardScaler`), selecao de atributos (`SelectKBest`) e balanceamento com SMOTE.
- **Modelos avaliados**: Regressao Logistica, Random Forest, Gradient Boosting, SVM e KNN, todos ajustados com `GridSearchCV`.
- **Ensemble**: Voting Classifier combinando os tres melhores modelos (votacao suave ponderada pelo F1 macro).
- **Artefatos gerados**: metricas consolidadas em CSV, matrizes de confusao em PNG e relatorio HTML estilizado. Um notebook Jupyter pode ser produzido automaticamente a partir do script.

## Estrutura
- `py/main.py` — script principal com funcoes reutilizaveis e interface de linha de comando.
- `output/` — pasta criada em tempo de execucao contendo:
  - `report.html`
  - `confusion_best.png`
  - `confusion_ensemble.png`
  - `metrics.csv`
- `notebooks/fraud_detection.ipynb` — notebook gerado sob demanda (ver proxima secao).

## Requisitos
1. Python 3.10 ou superior.
2. Disponibilize `creditcard - menor balanceado.csv` na raiz do projeto.
3. Instale as dependencias (user install recomendado):

```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib jinja2 nbformat
```

## Como executar
No diretório `TrabalhoPy` rode:

```bash
python py/main.py
```

Sem argumentos adicionais o script exibe um menu interativo:
```
1 - Treinamento CLI (pipeline)
2 - Gerar notebook Jupyter
3 - Executar pipeline e gerar notebook
```

- **Opcao 1**: executa o pipeline completo e gera os artefatos em `output/`, abrindo o HTML ao final.
- **Opcao 2**: apenas cria/atualiza `notebooks/fraud_detection.ipynb` (o notebook, ao ser executado, chama o mesmo pipeline).
- **Opcao 3**: executa o pipeline e, na sequencia, atualiza o notebook.

### Parametros uteis
- `--mode {pipeline,notebook,both}`: pula o menu interativo e escolhe o modo diretamente.
- `--dataset CAMINHO`: usa outro CSV no mesmo formato.
- `--output PASTA`: define pasta de saida (padrao `output/`).
- `--no-html`: desativa geracao e abertura do `report.html`.

Exemplo executando somente o notebook (sem HTML):

```bash
python py/main.py --mode notebook --no-html
```

## Resultados resumidos (execucao de referencia)
- Melhor modelo individual: Regressao Logistica (F1 macro de aproximadamente 0.9415).
- Ensemble: desempenho equivalente ao melhor modelo individual, mantendo alto recall para a classe fraude.
- Consulte `output/metrics.csv` e `output/report.html` para detalhes, matrizes de confusao e relatorios textuais.

## Personalizacoes sugeridas
- Ajustar os grids em `build_model_configs()` para ampliar a busca de hiperparametros.
- Incluir novos algoritmos (por exemplo XGBoost ou LightGBM) seguindo o padrao de pipeline.
- Estender o notebook com exploracao adicional ou visualizacoes complementares.

---
Trabalho realizado para estudo de detecao de fraudes em cartoes, contemplando balanceamento de dados, selecao de atributos, tuning de hiperparametros e ensembles.
