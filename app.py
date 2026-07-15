import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import io
import time
import zipfile
from PIL import Image

# =========================================================================
# CONFIGURAÇÃO / CALIBRAÇÃO 
# =========================================================================
HSV_RANGES = {
    "azul": [(np.array([90, 40, 40]), np.array([140, 255, 255]))],
    "verde": [(np.array([35, 40, 40]), np.array([85, 255, 255]))],
    "amarelo": [(np.array([15, 60, 60]), np.array([34, 255, 255]))],
}

AREA_MINIMA = 15
MIN_DISTANCE = 7

# =========================================================================
# FUNÇÕES DE PROCESSAMENTO
# =========================================================================
def criar_mascara(hsv_img, faixas):
    mask_total = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for baixo, alto in faixas:
        mask = cv2.inRange(hsv_img, baixo, alto)
        mask_total = cv2.bitwise_or(mask_total, mask)
    return mask_total

def limpar_mascara(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def contar_objetos(mask, area_minima=AREA_MINIMA, min_distance=MIN_DISTANCE):
    mask = limpar_mascara(mask)
    dist = ndi.distance_transform_edt(mask)

    if dist.max() == 0:
        return 0, np.zeros_like(mask, dtype=np.int32)

    coords = peak_local_max(dist, min_distance=min_distance, labels=mask)
    peaks_mask = np.zeros(dist.shape, dtype=bool)
    peaks_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks_mask)

    labels = watershed(-dist, markers, mask=mask)

    numero_celulas = 0
    labels_filtrados = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        area = np.sum(labels == label_id)
        if area >= area_minima:
            numero_celulas += 1
            labels_filtrados[labels == label_id] = label_id

    return numero_celulas, labels_filtrados

def desenhar_contornos(img, labels, cor):
    saida = img.copy()
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask_obj = np.uint8(labels == label_id) * 255
        contornos, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(saida, contornos, -1, cor, 1)
    return saida

def processar_imagem_na_memoria(file_bytes, nome_arquivo):
    """Lê a imagem (suporta TIFF), processa e desenha os resultados nela.

    Retorna: (relatorio, saida_rgb, saida_bgr, original_rgb, mensagem_erro)
    Em caso de falha, relatorio/saida_rgb/saida_bgr/original_rgb vêm como None
    e mensagem_erro traz o motivo.
    """

    # 0. Validação básica do arquivo antes de tentar processar
    TAMANHO_MINIMO_BYTES = 100  # arquivos válidos de imagem são bem maiores que isso
    if not file_bytes or len(file_bytes) < TAMANHO_MINIMO_BYTES:
        return None, None, None, None, "Arquivo vazio ou corrompido (tamanho insuficiente)."

    # 1. Leitura robusta usando PIL (Pillow) para suportar TIFFs e outras imagens
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()  # verifica integridade básica do arquivo
        # verify() invalida o objeto para leitura de pixels, então reabrimos
        pil_img = Image.open(io.BytesIO(file_bytes))
        # Converter para RGB caso seja paleta, escala de cinza ou RGBA
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        original_rgb = np.array(pil_img)
        # Converter para formato BGR do OpenCV
        img = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, None, None, f"Não foi possível ler a imagem ({e})."

    if img is None or img.size == 0:
        return None, None, None, None, "Imagem lida está vazia."

    # 2. Processamento (Máscaras e Contagem)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_azul = criar_mascara(hsv, HSV_RANGES["azul"])
    mask_verde = criar_mascara(hsv, HSV_RANGES["verde"])
    mask_amarelo = criar_mascara(hsv, HSV_RANGES["amarelo"])

    n_azul, lab_azul = contar_objetos(mask_azul)
    n_verde, lab_verde = contar_objetos(mask_verde)
    n_amarelo, lab_amarelo = contar_objetos(mask_amarelo)

    total_celulas = n_azul if n_azul > 0 else (n_verde + n_amarelo)
    viabilidade = (n_verde / total_celulas * 100) if total_celulas > 0 else 0.0

    # 3. Desenhar contornos
    saida = img.copy()
    saida = desenhar_contornos(saida, lab_azul, (255, 0, 0))     # azul (BGR)
    saida = desenhar_contornos(saida, lab_verde, (0, 255, 0))    # verde
    saida = desenhar_contornos(saida, lab_amarelo, (0, 255, 255))# amarelo

    relatorio = {
        "Arquivo": nome_arquivo,
        "Total (Azul)": n_azul,
        "Viáveis (Verde)": n_verde,
        "Mortas (Amarelo)": n_amarelo,
        "Base de Cálculo": total_celulas,
        "Viabilidade (%)": round(viabilidade, 2),
    }

    # 4. Escrever resultados em cima da imagem (Carimbo)
    texto_linhas = [
        f"Arquivo: {nome_arquivo}",
        f"Viabilidade: {relatorio['Viabilidade (%)']}%",
        f"Total: {relatorio['Base de Cálculo']}",
        f"Viáveis: {relatorio['Viáveis (Verde)']}",
        f"Mortas: {relatorio['Mortas (Amarelo)']}"
    ]
    
    y0, dy = 40, 40 # Posição inicial Y e espaçamento
    for i, linha in enumerate(texto_linhas):
        y = y0 + i * dy
        # Desenha borda preta para dar contraste em fundos claros
        cv2.putText(saida, linha, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        # Desenha texto branco por cima
        cv2.putText(saida, linha, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Converter de volta para RGB para exibir no Streamlit corretamente
    saida_rgb = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
    
    return relatorio, saida_rgb, saida, original_rgb, None


# =========================================================================
# INTERFACE STREAMLIT
# =========================================================================
st.set_page_config(page_title="ViaCell", layout="wide")

st.title("🔬 ViaCell")
st.markdown("Faça o upload de múltiplas imagens de fluorescência para análise em lote (Suporta JPG, PNG, BMP, TIF/TIFF).")

# Componente de upload múltiplo
arquivos_upados = st.file_uploader(
    "Arraste as imagens ou clique para procurar", 
    type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"], 
    accept_multiple_files=True
)

# Precisamos guardar os dados no session_state para que não sejam apagados
# quando o usuário interage com a navegação lateral (o que causa um "rerun" do script)
if "zip_gerado" not in st.session_state:
    st.session_state.zip_gerado = None
if "resultados" not in st.session_state:
    st.session_state.resultados = None
if "imagens_processadas" not in st.session_state:
    st.session_state.imagens_processadas = None  # nome_arquivo -> dict com dados de exibição

if arquivos_upados:
    if st.button("Processar Imagens", type="primary"):
        st.write("---")
        
        resultados = []
        imagens_para_salvar = []       # Lista para guardar as imagens geradas (para o ZIP)
        imagens_processadas = {}       # nome_arquivo -> {relatorio, processada_rgb, original_rgb}

        total_arquivos = len(arquivos_upados)
        tempos_processamento = []  # guarda a duração de cada imagem já processada

        barra_progresso = st.progress(0.0)
        texto_status = st.empty()

        st.markdown("#### Imagens processadas")
        galeria_ao_vivo = st.container()

        # Processa as imagens uma a uma; cada resultado fica disponível
        # para visualização imediatamente (expander), sem esperar o lote todo
        for idx, arquivo in enumerate(arquivos_upados):
            inicio = time.time()
            relatorio, img_rgb_tela, img_bgr_salvar, original_rgb, erro = processar_imagem_na_memoria(
                arquivo.read(), arquivo.name
            )
            duracao = time.time() - inicio
            tempos_processamento.append(duracao)

            if erro is not None:
                st.error(f"Erro ao processar '{arquivo.name}': {erro}")
            else:
                resultados.append(relatorio)
                imagens_para_salvar.append((arquivo.name, img_bgr_salvar))
                imagens_processadas[arquivo.name] = {
                    "relatorio": relatorio,
                    "processada_rgb": img_rgb_tela,
                    "original_rgb": original_rgb,
                }

                # Salva incrementalmente no session_state: se o usuário interagir
                # com algo antes do lote terminar, o que já foi processado não se perde
                st.session_state.resultados = list(resultados)
                st.session_state.imagens_processadas = dict(imagens_processadas)

                # Disponibiliza esta imagem para visualização imediatamente,
                # mesmo com as demais ainda em processamento
                with galeria_ao_vivo:
                    with st.expander(
                        f"✅ {arquivo.name} — Viabilidade: {relatorio['Viabilidade (%)']}%",
                        expanded=False,
                    ):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.write(f"**Total de Células:** {relatorio['Total (Azul)']}")
                            st.write(f"**Vivas:** {relatorio['Viáveis (Verde)']}")
                            st.write(f"**Mortas:** {relatorio['Mortas (Amarelo)']}")
                        with col_b:
                            st.image(img_rgb_tela, use_container_width=True)

            # Atualiza barra de progresso e tempo estimado restante
            processados = idx + 1
            progresso = processados / total_arquivos
            tempo_medio = sum(tempos_processamento) / len(tempos_processamento)
            restantes = total_arquivos - processados
            tempo_restante = tempo_medio * restantes

            if tempo_restante >= 60:
                tempo_restante_str = f"{tempo_restante / 60:.1f} min"
            else:
                tempo_restante_str = f"{tempo_restante:.0f} s"

            barra_progresso.progress(progresso)
            if restantes > 0:
                texto_status.info(
                    f"Processadas {processados}/{total_arquivos} imagens · "
                    f"Tempo estimado restante: {tempo_restante_str}"
                )
            else:
                texto_status.success(f"Concluído! {processados}/{total_arquivos} imagens processadas.")

        # Garante que o estado final esteja salvo (redundante, mas seguro)
        st.session_state.resultados = resultados
        st.session_state.imagens_processadas = imagens_processadas

        # -------------------------------------------------------------
        # Geração do Arquivo ZIP com os Relatórios e Imagens Anotadas
        # -------------------------------------------------------------
        if resultados:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # 1. Salvar o CSV de relatórios no ZIP
                df_resultados_zip = pd.DataFrame(resultados)
                csv_bytes = df_resultados_zip.to_csv(index=False).encode('utf-8')
                zip_file.writestr("relatorio_consolidado.csv", csv_bytes)

                # 2. Salvar cada imagem com os resultados desenhados
                for nome_original, img_bgr in imagens_para_salvar:
                    # Garantir a extensão PNG na saída (melhor compressão e leitura)
                    nome_base = nome_original.rsplit('.', 1)[0]
                    nome_arquivo_zip = f"{nome_base}_resultado.png"
                    
                    # Codificar imagem em memória
                    sucesso, buffer_img = cv2.imencode('.png', img_bgr)
                    if sucesso:
                        zip_file.writestr(nome_arquivo_zip, buffer_img.tobytes())

            st.session_state.zip_gerado = zip_buffer.getvalue()

# -------------------------------------------------------------
# Exibição dos resultados (lida do session_state, sobrevive a reruns)
# -------------------------------------------------------------
imagens_processadas = st.session_state.get("imagens_processadas")
resultados = st.session_state.get("resultados")

if imagens_processadas:
    st.write("---")
    st.markdown("#### Navegador de resultados")
    nomes_disponiveis = list(imagens_processadas.keys())

    with st.sidebar:
        st.markdown("### Resultados")
        imagem_selecionada = st.radio(
            "Selecione uma imagem para visualizar",
            options=nomes_disponiveis,
            label_visibility="collapsed",
        )

    dados_sel = imagens_processadas[imagem_selecionada]
    relatorio_sel = dados_sel["relatorio"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Resultados")
        st.metric("Viabilidade", f"{relatorio_sel['Viabilidade (%)']}%")
        st.write(f"**Total de Células:** {relatorio_sel['Total (Azul)']}")
        st.write(f"**Vivas:** {relatorio_sel['Viáveis (Verde)']}")
        st.write(f"**Mortas:** {relatorio_sel['Mortas (Amarelo)']}")

        # Aviso quando a marcação azul (total) parece subestimada em relação a verde+amarelo
        n_azul = relatorio_sel['Total (Azul)']
        n_verde_amarelo = relatorio_sel['Viáveis (Verde)'] + relatorio_sel['Mortas (Amarelo)']
        if n_azul > 0 and n_verde_amarelo > 0 and n_azul < 0.7 * n_verde_amarelo:
            st.warning(
                "⚠️ A contagem em azul (usada como total) está bem menor que "
                "vivas + mortas somadas. Isso pode indicar marcação azul fraca "
                "nesta imagem — considere revisar a calibração ou a imagem original."
            )

        modo_visualizacao = st.radio(
            "Visualizar",
            options=["Resultado processado", "Imagem original"],
            horizontal=True,
        )

    with col2:
        if modo_visualizacao == "Imagem original":
            st.image(
                dados_sel["original_rgb"],
                caption=f"Imagem original - {imagem_selecionada}",
                use_container_width=True,
            )
        else:
            st.image(
                dados_sel["processada_rgb"],
                caption=f"Contornos e Resultados - {imagem_selecionada}",
                use_container_width=True,
            )

if resultados:
    st.write("---")
    st.subheader("📊 Relatório")
    df_resultados = pd.DataFrame(resultados)
    st.dataframe(df_resultados, use_container_width=True)

# Exibir o botão de Download caso o ZIP já tenha sido gerado
if st.session_state.get("zip_gerado"):
    st.markdown("### 📥 Download dos Resultados")
    st.download_button(
        label="Baixar Todas as Imagens + Planilha (ZIP)",
        data=st.session_state.zip_gerado,
        file_name="ViaCell_Resultados.zip",
        mime="application/zip"
    )