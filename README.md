# Detecção de Fraudes em Cartões de Crédito

Projeto desenvolvido a partir do conjunto de dados fornecido em `creditcard - menor balanceado.csv` e das diretrizes do documento _Trab. Machine Learning.pdf_ (ambos devem ser adicionados manualmente à pasta do projeto). O objetivo é construir, comparar e combinar modelos de Machine Learning capazes de identificar transações fraudulentas em um cenário desbalanceado.

## Visão Geral
- **Pré-processamento**: padronização (`StandardScaler`), seleção de atributos (`SelectKBest`) e balanceamento com SMOTE.
- **Modelos treinados**: Regressão Logística, Random Forest, Gradient Boosting, SVM e KNN (todos com _GridSearchCV_ e validação estratificada).
- **Ensemble**: Voting Classifier combinando os três melhores modelos individuais (voto suave ponderado pelo F1 macro).
- **Relatórios**: métricas consolidadas em CSV, matrizes de confusão em PNG e dashboard HTML estilizado.

## Estrutura dos Arquivos
- `py/main.py` — script principal que executa todo o pipeline.
- `output/` — pasta gerada automaticamente com:
  - `report.html` — relatório interativo resumindo resultados.
  - `confusion_best.png` — matriz de confusão do melhor modelo individual.
  - `confusion_ensemble.png` — matriz de confusão do ensemble.
  - `metrics.csv` — tabela com métricas de validação cruzada e de teste.

## Requisitos
Certifique-se de possuir Python 3.10+ instalado e de disponibilizar o arquivo `creditcard - menor balanceado.csv` na raiz do projeto (o dataset não é versionado neste repositório). Instale as dependências (user install recomendado em ambientes sem privilégios):

```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib jinja2
```

## Como Executar
No diretório `TrabalhoPy`, rode:

```bash
python py/main.py
```

O script irá:
1. Ler o CSV e separar treino/teste estratificado (20% para teste).
2. Ajustar e avaliar todos os modelos definidos.
3. Construir o ensemble com os três melhores resultados.
4. Gerar métricas e arquivos de saída na pasta `output/`.

### Parâmetros úteis
- `--dataset CAMINHO` — especifica outro CSV no mesmo formato.
- `--output PASTA` — muda o diretório onde os resultados serão salvos (padrão `output/`).
- `--no-html` — pula apenas a geração do `report.html` (demais arquivos são produzidos).

Exemplo utilizando dataset alternativo e suprimindo o HTML:

```bash
python py/main.py --dataset dados.csv --output resultados --no-html
```

## Resultados Obtidos (amostra)
- Melhor modelo individual (Regressão Logística): F1 Macro ≈ 0.9415, Recall Macro ≈ 0.9278, Acurácia ≈ 0.9545.
- Ensemble (Voting Classifier): desempenho equivalente ao melhor modelo individual, mantendo recall elevado para a classe de fraude.
- Consulte `output/metrics.csv` e `output/report.html` para detalhes completos, comparações, matrizes de confusão e relatórios por classe.

## Personalizações Sugeridas
- Ajustar os grids de hiperparâmetros em `build_model_configs()` para expandir a busca.
- Incluir novos classificadores (ex.: XGBoost, LightGBM) seguindo o mesmo padrão de _pipeline_.
- Acrescentar novas visualizações no HTML (ex.: importância de features) aproveitando o template Jinja2 existente.

---
Trabalho desenvolvido para o estudo de detecção de fraudes com foco em dados desbalanceados e comparação de técnicas de balanceamento, seleção de atributos, ajuste fino e ensemble.
