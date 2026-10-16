# Análise inicial — ML Group 04

Revisão de 9 de outubro de 2026. Foram inspecionados o repositório clonado, os quatro CSV, os dois enunciados incluídos em `Project.zip` e os oito notebooks das semanas 2–6 indicados pelo utilizador. Os documentos foram tratados como material de referência e requisitos académicos; esta revisão não constitui a execução de todos os exercícios das aulas.

## Objetivo e estado atual

Prever `repeat_purchase_90d`: se o cliente faz outra compra nos 90 dias seguintes à compra atual. É classificação binária; cada linha representa uma ocasião de compra, não um cliente.

O repositório, no commit `4dc0927`, contém dados, dependências, README e um notebook inicial de EDA. As células do notebook original não têm execuções guardadas. Ainda não existem limpeza implementada, seleção de atributos, modelos, avaliação, otimização ou geração de submissões. É um ponto de partida, não uma entrega completa.

## Entregas descritas nos enunciados

| Entrega | Prazo indicado | Conteúdo |
|---|---|---|
| Homework | 26/10/2026, 17:59 | Notebook ou ZIP de notebooks e PDF de 2 páginas; nomes `Homework_Group04` |
| Projeto final | 22/12/2026, 17:59 | Notebook ou ZIP completo, incluindo deployment e solução escolhida no Kaggle |

No homework: exploração (3 valores), pré-processamento (5), seleção de atributos (3), pelo menos um modelo e avaliação/predições (4), PDF da estrutura e justificação (5). Há um ponto extra para os cinco melhores grupos no Kaggle, com nota limitada a 20.

No projeto final: estrutura (4 valores), EDA/preparação (4), comparação/otimização (7), secção aberta (4), deployment (1), e um ponto extra por disponibilização pública no GitHub. A comparação exige pelo menos **cinco algoritmos abordados nas aulas**. O notebook deve incluir contribuições dos membros, abstract de 200–300 palavras, necessidade de negócio, EDA/preparação, classificação/otimização, secção aberta e deployment.

O enunciado proíbe Lazy Predict e ferramentas AutoML semelhantes, exige execução integral dos notebooks e participação no Kaggle para pontuar essa componente. A métrica oficial do Kaggle não aparece nos PDFs: deve ser confirmada na página da competição antes de orientar a otimização. O `sample_submission.csv` contém previsões de baseline; não contém o target verdadeiro do teste.

## Auditoria dos dados

Os quatro CSV têm conteúdo tabular igual ao ZIP. Os ficheiros diferem ao nível dos bytes, mas as células CSV comparadas são iguais.

| Ficheiro | Linhas | Colunas | Observação |
|---|---:|---:|---|
| train.csv | 6 534 | 39 | Inclui target |
| test.csv | 986 | 38 | Mesmos preditores, sem target |
| clientes.csv | 1 143 | 13 | `customer_id` único |
| sample_submission.csv | 986 | 2 | `ID`, `repeat_purchase_90d` |

- Target de treino: 3 654 positivos (55,92%) e 2 880 negativos (44,08%). A classe positiva é maioritária; convém reportar F1 com uma baseline e métricas complementares.
- Treino: 610 clientes. Teste: 206 clientes, dos quais 184 também aparecem no treino. Compras repetidas do mesmo cliente são legítimas e não devem ser removidas apenas por partilharem `customer_id`.
- Todas as chaves de clientes presentes em treino/teste existem em `clientes.csv`; uma junção deve ser validada como `many_to_one`, mantendo número e ordem das linhas.
- Datas válidas de treino: 11/07/2015–15/02/2025. Teste: 24/05/2025–06/10/2026. O enunciado diz que as datas são anonimizadas, preservando ordem e intervalos; não assumir que feriados reais ou contexto económico dessas datas são aplicáveis.
- Há **65 linhas de treino com todos os preditores vazios, exceto `ID` e `customer_id`**, mas com target presente. Devem ser identificadas antes de definir a limpeza e a validação temporal. Comparar e justificar exclusão ou tratamento; não remover silenciosamente.
- Há **10 duplicados adicionais quando se exclui apenas `ID`**, todos em linhas sem data. Quando se excluem também os targets, há 12 repetições adicionais e **dois grupos com targets contraditórios**. Como faltam os atributos da compra, a igualdade do conteúdo disponível não prova que eram a mesma compra: exige inspeção; não é suficiente usar `train.duplicated()` com `ID` incluído. Entre linhas com data, não há duplicados de `customer_id` + `purchase_date`.
- `days_since_previous_purchase` contém **16 valores negativos** (mínimo −4): incompatíveis com a definição de recência. Inspecionar e definir regra reproduzível, por exemplo converter valores impossíveis em ausentes, com justificação.
- Histórico em falta pode significar ausência de compras anteriores. Distinguir este caso de falhas de qualidade; considerar um indicador de cliente sem histórico.
- `customer_type` está ausente nas 1 143 linhas de clientes. Não traz sinal observado e não deve ser aceite automaticamente como atributo útil.
- Existem muitos ausentes em geografia e moradas. Moradas, ruas e nomes de empresas podem ter cardinalidade elevada; avaliar custo e generalização antes de os incluir.
- `random_noise` é explicitamente uma variável a avaliar no enunciado. Testar a sua exclusão através de comparação em validação, sem assumir utilidade por ser numérica.

## Revisão do notebook existente

O notebook cobre leitura, distribuição do target, estrutura, ausentes, estatísticas e compatibilidade de colunas. Já distingue identificadores e reconhece que o teste é posterior ao treino.

Pontos a completar:

1. `DATA_DIR = Path('data')` depende do diretório de trabalho. Funciona na raiz do repositório, mas a execução com diretório de trabalho `notebooks/` procura dados no sítio errado. Tornar a descoberta da raiz robusta antes da entrega; o Jupyter pode iniciar o kernel na pasta do notebook.
2. A lista `numeric_columns` ainda inclui `ID` e `customer_id`. Não é usada para treinar neste notebook, mas deve ser corrigida antes de ser reutilizada num pipeline.
3. A verificação de duplicados inclui `ID`, escondendo as cópias descritas acima. Acrescentar auditoria por conteúdo e por ocasião de compra, distinguindo cópias de compras legítimas.
4. Acrescentar observações escritas que liguem os resultados às decisões de preparação, incluindo linhas vazias, recência negativa, campos sem informação e possíveis redundâncias monetárias.
5. Incorporar no notebook a validação de IDs, unicidade, cobertura da junção e alinhamento da submissão com o template. Na revisão, os IDs de treino/teste são únicos e os IDs do teste estão alinhados com o template, incluindo a ordem.

## Relação com os materiais das aulas

| Material | Aplicação ao projeto |
|---|---|
| Semana 2 — pipeline | Organizar necessidade de negócio → dados → preparação → modelos → otimização → avaliação → deployment; guardar pipeline final |
| Semana 3 — classificação | Auditar valores impossíveis e duplicados; comparar imputação e tratamento de extremos em vez de escolher por hábito |
| Semana 3 — regressão | Aproveitar método de diagnóstico e comparação; os resultados e regras do exemplo de automóveis não se transferem diretamente |
| Semana 4 — classificação | Encoding, scaling, atributos derivados e seleção; aprender todas as escolhas dependentes dos dados dentro dos folds |
| Semana 4 — regressão | Referência adicional sobre transformações, redundância e seleção; o nosso target continua binário |
| Semana 5 — model selection | Pipeline dentro da pesquisa; distinguir seleção de configurações e estimativa do desempenho; justificar folds temporais ou agrupados |
| Semana 5 — performance measures | Baselines, matriz de confusão, precision, recall, F1, ROC-AUC e efeito do threshold |
| Semana 6 — linear/logistic | Regressão logística como baseline interpretável; regularização e comparação com DummyClassifier |

Os notebooks das aulas usam dados e recursos externos próprios. Foram inspecionados, mas não executados neste projeto. Não copiar os números de desempenho ou as escolhas desses exemplos como resultados deste dataset.

## Sequência de trabalho proposta

1. Completar a auditoria e definir regras determinísticas de limpeza, preservando os CSV originais.
2. Fixar validação temporal em `train.csv`, com períodos posteriores reservados para avaliação. Como o target olha 90 dias para a frente, garantir que as etiquetas das compras usadas para treinar já estariam disponíveis no início do período de validação; testar uma separação de 90 dias e documentar a perda de amostra. Tratar explicitamente as linhas sem data.
3. A repetição de clientes não implica automaticamente usar GroupKFold: prever compras futuras de clientes conhecidos e generalizar para clientes novos são perguntas diferentes. Usar a primeira como cenário principal, de acordo com o teste posterior, e considerar a segunda como análise adicional.
4. Implementar baseline com DummyClassifier, regressão logística e árvore de decisão. Numéricos: imputação e scaling quando adequado; categorias: imputação e one-hot com categorias novas toleradas. Excluir `ID` e começar sem `customer_id` como preditor.
5. Comparar integração de `clientes.csv`, atributos derivados, alternativas de imputação e uma estratégia explícita de seleção, usando as mesmas divisões e métricas. Ajustar imputação, encoding, scaling e seleção apenas nas linhas de treino de cada fold.
6. Para o homework, gerar previsões para os 986 IDs e escrever o PDF de 2 páginas com diagrama e justificação apoiada nos resultados.
7. Para o final, comparar pelo menos cinco algoritmos ensinados, otimizar os candidatos e desenvolver uma secção aberta. Uma opção coerente é uma ablação de atributos transacionais, histórico e informação de clientes, com análise de erros em clientes novos e conhecidos.
8. Guardar o pipeline final e produzir a submissão respeitando exatamente os IDs e o formato oficial. Confirmar na competição se são pedidos rótulos ou probabilidades.

Nenhum modelo foi treinado nesta revisão, nem foi feita submissão no Kaggle. As propostas acima são próximos passos, não resultados experimentais.

## Ambiente e verificação

Foi criada uma `.venv` local com as dependências de `requirements.txt` e `pypdf` para leitura dos enunciados. Para usar no macOS, a partir da raiz do repositório:

```sh
source .venv/bin/activate
python -m jupyter lab
```

Todas as células de código do notebook existente executaram sem erros a partir da raiz do repositório. O registo está em `outputs/review/notebook_execution.txt` (pasta ignorada pelo Git). A execução feita na revisão usa Python e `display` do IPython, com backend gráfico não interativo; não substitui a verificação final num kernel Jupyter a partir de um ambiente limpo.
