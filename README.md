# 🔬 ViaCell

Aplicativo web para contagem automática de células em imagens de fluorescência (ensaios de viabilidade celular / live-dead assay), com processamento em lote.

O app identifica células marcadas por cor:
- **Azul** — todas as células (núcleos, ex: DAPI/Hoechst)
- **Verde** — células vivas
- **Amarelo/Laranja** — células mortas

E calcula automaticamente a **viabilidade celular**:

```
viabilidade (%) = (nº células vivas / nº total de células) × 100
```

## ✨ Funcionalidades

- Upload de múltiplas imagens simultaneamente (JPG, PNG, BMP, TIF/TIFF)
- Processamento em lote com barra de progresso e tempo estimado restante
- Visualização progressiva: cada imagem fica disponível assim que termina de processar, sem precisar esperar o lote inteiro
- Navegação lateral entre os resultados de cada imagem
- Alternância entre imagem original e imagem processada (com contornos)
- Alerta automático quando a contagem de células totais (azul) parece subestimada
- Relatório consolidado em tabela
- Download de um `.zip` com todas as imagens anotadas + relatório em `.csv`

## 🧠 Como funciona

1. A imagem é convertida para o espaço de cor HSV (mais fácil de isolar cores).
2. São criadas máscaras binárias para azul, verde e amarelo com base em faixas de matiz (Hue) predefinidas.
3. Ruído é removido com operações morfológicas (abertura/fechamento).
4. O algoritmo **watershed** separa células que estão coladas/sobrepostas, usando a transformada de distância para localizar o centro de cada célula.
5. Cada célula detectada é contada, desde que tenha uma área mínima (para descartar ruído).
6. É gerada uma imagem de saída com os contornos das células detectadas, além do relatório com as contagens e a viabilidade calculada.

> **Nota:** as faixas de cor (HSV) usadas na detecção foram ajustadas para um conjunto de imagens de exemplo. Cada experimento/microscópio pode gerar cores um pouco diferentes de brilho e saturação — se a contagem parecer incorreta, os valores em `HSV_RANGES`, `AREA_MINIMA` e `MIN_DISTANCE` no início de `app.py` podem precisar de ajuste.

## 🚀 Instalação

**1. Clone o repositório**
```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
```

**2. (Recomendado) Crie um ambiente virtual**
```bash
python3 -m venv venv
```

Ative o ambiente:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

> Em sistemas Linux mais restritos (ex: Ubuntu 24+), pode ser necessário adicionar `--break-system-packages`:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```

## ▶️ Uso

```bash
streamlit run app.py
```

O app abrirá automaticamente no navegador (geralmente em `http://localhost:8501`).

1. Faça upload de uma ou mais imagens.
2. Clique em **"Processar Imagens"**.
3. Acompanhe o progresso e visualize os resultados conforme ficam prontos.
4. Baixe o `.zip` com as imagens anotadas e o relatório consolidado em `.csv`.

## 📁 Estrutura do projeto

```
.
├── app.py              # Aplicativo Streamlit principal
├── requirements.txt     # Dependências do projeto
├── .gitignore
└── README.md
```

## 🛠️ Tecnologias

- [Streamlit](https://streamlit.io/) — interface web
- [OpenCV](https://opencv.org/) — processamento de imagem
- [scikit-image](https://scikit-image.org/) — segmentação (watershed)
- [SciPy](https://scipy.org/) — transformada de distância
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) — manipulação de dados
- [Pillow](https://python-pillow.org/) — leitura de imagens (incluindo TIFF)

## 📄 Licença

Este projeto está licenciado sob os termos da licença MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.