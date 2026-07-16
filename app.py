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
AREA_MINIMA_CELULA = 15     # área mínima (px) para núcleos/células (azul e verde)
AREA_MINIMA_MORTA = 4       # área mínima (px) para pontos de morte (amarelo, tende a ser pequeno)
MIN_DISTANCE_CELULA = 7     # distância mínima entre picos no watershed (azul/verde)
MIN_DISTANCE_MORTA = 3      # distância mínima entre picos no watershed (amarelo)
RUIDO_AREA_MINIMA_PX = 6    # qualquer componente conexo menor que isso é ruído de sensor, nunca célula
FRACAO_MINIMA_AMARELA = 0.15  # fração mínima da região dilatada do núcleo que precisa estar
                               # marcada na máscara amarela limpa para classificar como morta
                               # (evita que manchas amarelas pequenas/concentradas sejam
                               # diluídas pela média da região inteira)
FRACAO_MINIMA_VERDE = 0.15    # mesmo critério, para o sinal verde (viabilidade)

# =========================================================================
# FUNÇÕES DE PROCESSAMENTO
# =========================================================================

def normalizar_robusto(canal_float, p_baixo=0.5, p_alto=99.5):
    """
    AJUSTE DE PRECISÃO: cv2.normalize com NORM_MINMAX usa o mínimo e o máximo
    BRUTOS do array pra definir a escala 0-255. Um único pixel saturado (hot
    pixel, reflexo, artefato de sensor/compressão) infla o máximo e comprime
    o resto da imagem numa faixa estreita de cinza — deslocando o threshold de
    Otsu de forma inconsistente entre imagens. Isso é parte de por que um BMP
    não-comprimido (mais outliers de sensor crus) e um JPEG (comprimido, mais
    "limpo") do mesmo tipo de amostra geram thresholds tão diferentes. Usar
    percentis (0.5 / 99.5 por padrão) em vez de min/max bruto faz com que
    alguns poucos pixels extremos não consigam mais dominar a escala inteira
    da imagem.
    """
    lo, hi = np.percentile(canal_float, [p_baixo, p_alto])
    if hi <= lo:
        lo, hi = float(canal_float.min()), float(canal_float.max())
        if hi <= lo:
            return np.zeros_like(canal_float, dtype=np.uint8)
    canal_clip = np.clip(canal_float, lo, hi)
    canal_norm = (canal_clip - lo) / (hi - lo) * 255.0
    return canal_norm.astype(np.uint8)


def estimar_kernel_tophat(shape):
    menor_lado = min(shape[:2])
    kernel = int(menor_lado * 0.025)
    kernel = kernel + 1 if kernel % 2 == 0 else kernel  # precisa ser ímpar
    return int(np.clip(kernel, 9, 61))

def corrigir_fundo(canal, kernel_size=None, aplicar_bilateral=True):
    """
    Corrige iluminação de fundo desigual usando top-hat morfológico.
    AJUSTE DE PRECISÃO: Median blur remove ruído tipo "sal e pimenta" do sensor
    (pontos isolados de 1-2px) e o Filtro Bilateral em seguida suaviza mantendo
    as bordas das células nítidas. A ordem importa: median primeiro elimina os
    outliers pontuais que o bilateral sozinho preserva.
    """
    # Median blur: mata ruído de pixel único sem borrar bordas reais.
    canal = cv2.medianBlur(canal, 3)

    if aplicar_bilateral:
        # Filtro bilateral: reduz ruído residual, preserva bordas cravadas.
        canal = cv2.bilateralFilter(canal, d=5, sigmaColor=50, sigmaSpace=50)

    if kernel_size is None:
        kernel_size = estimar_kernel_tophat(canal.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(canal, cv2.MORPH_TOPHAT, kernel)
    return tophat


def limiar_otsu_com_ajuste(canal, bias=0):
    canal_8u = cv2.normalize(canal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canal_suave = cv2.GaussianBlur(canal_8u, (5, 5), 0)
    valor_otsu, _ = cv2.threshold(canal_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    valor_final = float(np.clip(valor_otsu - bias, 1, 254))
    _, mask = cv2.threshold(canal_suave, valor_final, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8), valor_final, canal_suave


def remover_ruido_pontual(mask, area_minima_px=RUIDO_AREA_MINIMA_PX):
    """
    AJUSTE DE PRECISÃO: remove componentes conexos minúsculos (specks de 1-5px)
    ANTES de qualquer estimativa de escala/watershed. Sem isso, ruído de sensor
    é contado como "célula" e ainda contamina a mediana de área usada pelo modo
    automático (escala fica pequena -> filtro de área mínima fica frouxo -> mais
    ruído passa -> loop vicioso). Isso resolve o problema na raiz.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    saida = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_minima_px:
            saida[labels == i] = 255
    return saida


def limpar_mascara(mask):
    # 1) Remove ruído pontual de sensor antes de qualquer outra operação.
    mask = remover_ruido_pontual(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 2) Preenche buracos internos: núcleos com centro mais escuro não podem
    #    virar "anéis" que o watershed interpreta como múltiplos objetos.
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    # 3) Segunda passada de remoção de ruído, agora pós-morfologia.
    mask = remover_ruido_pontual(mask)
    return mask


def estimar_escala_objeto(mask):
    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
        return None
    areas = ndi.sum(np.ones_like(labels), labels, index=range(1, n_labels))
    areas = np.asarray(areas)
    areas = areas[areas > 0]
    if areas.size == 0:
        return None
    return float(np.median(areas))


def contar_objetos(mask, area_minima=None, area_maxima=None, min_circularidade=0.0, 
                   min_distance=None, auto=True, fator_area_min=0.12, fator_distancia=0.6,
                   area_minima_piso=RUIDO_AREA_MINIMA_PX, distancia_piso=2, distancia_teto=25):
    """
    AJUSTE DE PRECISÃO: Adicionado filtro de circularidade e área máxima 
    para remover sujeiras e aglomerados que o watershed não separou direito.
    """
    mask = limpar_mascara(mask)

    if auto:
        escala = estimar_escala_objeto(mask)
        if escala is not None and escala > 0:
            raio_equivalente = np.sqrt(escala / np.pi)
            min_distance = int(np.clip(raio_equivalente * fator_distancia, distancia_piso, distancia_teto))
            area_minima = max(area_minima_piso, escala * fator_area_min)
        else:
            min_distance = min_distance or distancia_piso
            area_minima = area_minima or area_minima_piso

    dist = ndi.distance_transform_edt(mask)

    if dist.max() == 0:
        return 0, np.zeros_like(mask, dtype=np.int32), {"area_minima": area_minima, "min_distance": min_distance}

    # AJUSTE DE PRECISÃO: a busca de picos usa uma versão suavizada do distance
    # transform, não o bruto. Sem isso, ruído pixel-a-pixel (mais presente em
    # imagens não-comprimidas, ex: BMP, que não recebem a suavização "de
    # fábrica" da compressão JPEG) cria múltiplos máximos locais espúrios
    # dentro de um único núcleo, fragmentando uma célula em 2-3 "células"
    # menores — o que infla a contagem total e corrompe a classificação
    # viável/morta (cada fragmento só vê parte do sinal verde/amarelo real).
    # O watershed em si continua usando o distance transform ORIGINAL (não
    # suavizado) como topografia, então os contornos finais não perdem
    # precisão — só a etapa de "quantos objetos existem aqui" fica mais
    # robusta a ruído.
    dist_suave = ndi.gaussian_filter(dist, sigma=1.0)
    coords = peak_local_max(dist_suave, min_distance=int(min_distance), labels=mask)
    peaks_mask = np.zeros(dist.shape, dtype=bool)
    peaks_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks_mask)

    labels = watershed(-dist, markers, mask=mask)

    numero_celulas = 0
    labels_filtrados = np.zeros_like(labels)
    
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        
        # Filtro de Área de Pixels Bruta
        area_pixel = np.sum(labels == label_id)
        if area_pixel < area_minima:
            continue
        if area_maxima is not None and area_pixel > area_maxima:
            continue

        # Filtro de Circularidade de Precisão
        if min_circularidade > 0.0:
            mask_obj = np.uint8(labels == label_id) * 255
            contornos, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos:
                continue
            
            c = contornos[0]
            area_contour = cv2.contourArea(c)
            perimetro = cv2.arcLength(c, True)
            
            if perimetro == 0:
                continue
                
            circularidade = (4 * np.pi * area_contour) / (perimetro * perimetro)
            if circularidade < min_circularidade:
                continue # Descarta detritos pontiagudos/linhas finas

        numero_celulas += 1
        labels_filtrados[labels == label_id] = label_id

    return numero_celulas, labels_filtrados, {"area_minima": round(area_minima, 1), "min_distance": min_distance}


def desenhar_contornos(img, labels, cor):
    saida = img.copy()
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask_obj = np.uint8(labels == label_id) * 255
        contornos, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(saida, contornos, -1, cor, 1)
    return saida


def segmentar_canal(canal_rgb_uint8, parametros, auto=True):
    corrigido = corrigir_fundo(canal_rgb_uint8, parametros.get("kernel_tophat"), parametros.get("aplicar_bilateral", True))
    mask, limiar_usado, _ = limiar_otsu_com_ajuste(corrigido, parametros.get("bias", 0))
    n_objetos, labels, params_usados = contar_objetos(
        mask, 
        auto=auto, 
        area_maxima=parametros.get("area_maxima"),
        min_circularidade=parametros.get("min_circularidade", 0.0)
    )
    return n_objetos, labels, mask, limiar_usado, params_usados


def classificar_nucleos(labels_azul, mask_verde, mask_amarelo,
                         raio_dilatacao=4, min_pixels_regiao=6):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (raio_dilatacao * 2 + 1,) * 2)

    ids = [l for l in np.unique(labels_azul) if l != 0]
    if not ids:
        vazio = np.zeros_like(labels_azul)
        return {
            "n_viaveis": 0, "n_mortas": 0, "n_indeterminadas": 0,
            "labels_viaveis": vazio, "labels_mortas": vazio, "labels_indeterminadas": vazio,
            "tabela_celulas": [],
        }

    n_viaveis = n_mortas = n_indeterminadas = 0
    labels_viaveis = np.zeros_like(labels_azul)
    labels_mortas = np.zeros_like(labels_azul)
    labels_indeterminadas = np.zeros_like(labels_azul)
    tabela_celulas = []

    for label_id in ids:
        obj_mask = np.uint8(labels_azul == label_id) * 255
        regiao = cv2.dilate(obj_mask, kernel) > 0
        area_regiao = int(regiao.sum())
        if area_regiao < min_pixels_regiao:
            continue

        # AJUSTE: ambos os canais (verde e amarelo) agora são as MÁSCARAS JÁ
        # LIMPAS (pós remoção de ruído pontual + abertura/fechamento morfológico
        # + preenchimento de buracos) — as mesmas que aparecem no debug visual.
        # Antes, essa função recebia o canal contínuo pré-limpeza (canal_suave),
        # o que fazia a classificação responder a ruído de sensor em vez da
        # máscara real que o usuário confere visualmente. Medimos a fração de
        # pixels "ligados" da máscara dentro da região dilatada do núcleo.
        frac_verde = float((mask_verde[regiao] > 127).mean())
        frac_amarela = float((mask_amarelo[regiao] > 127).mean())
        pos_verde = frac_verde >= FRACAO_MINIMA_VERDE
        pos_amarelo = frac_amarela >= FRACAO_MINIMA_AMARELA

        # AJUSTE DE PRECISÃO: o marcador de morte (rompimento de membrana) é o
        # critério biologicamente dominante em ensaios live/dead — se a célula
        # está acima do seu próprio limiar de amarelo, ela é morta, mesmo que
        # ainda carregue algum sinal verde residual. A versão anterior comparava
        # (amarelo - limiar) contra (verde - limiar) em unidades absolutas, e
        # como o canal verde tem escala de intensidade naturalmente maior, o
        # verde quase sempre "vencia" essa disputa — subcontando mortas mesmo
        # com sinal amarelo claramente acima do limiar.
        if pos_amarelo:
            classe = "morta"
            n_mortas += 1
            labels_mortas[labels_azul == label_id] = label_id
        elif pos_verde:
            classe = "viavel"
            n_viaveis += 1
            labels_viaveis[labels_azul == label_id] = label_id
        else:
            classe = "indeterminada"
            n_indeterminadas += 1
            labels_indeterminadas[labels_azul == label_id] = label_id

        tabela_celulas.append({
            "celula_id": int(label_id),
            "area_px": area_regiao,
            "cobertura_verde_pct": round(frac_verde * 100, 1),
            "cobertura_amarela_pct": round(frac_amarela * 100, 1),
            "classificacao": classe,
        })

    return {
        "n_viaveis": n_viaveis,
        "n_mortas": n_mortas,
        "n_indeterminadas": n_indeterminadas,
        "labels_viaveis": labels_viaveis,
        "labels_mortas": labels_mortas,
        "labels_indeterminadas": labels_indeterminadas,
        "tabela_celulas": tabela_celulas,
    }


def processar_imagem_na_memoria(file_bytes, nome_arquivo, parametros):
    TAMANHO_MINIMO_BYTES = 100
    if not file_bytes or len(file_bytes) < TAMANHO_MINIMO_BYTES:
        return None, None, None, None, None, "Arquivo vazio ou corrompido."

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(file_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        original_rgb = np.array(pil_img)
        img = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, None, None, None, f"Não foi possível ler a imagem ({e})."

    r = original_rgb[..., 0].astype(np.float32)
    g = original_rgb[..., 1].astype(np.float32)
    b = original_rgb[..., 2].astype(np.float32)

    kernel_tophat = parametros.get("kernel_tophat")
    aplicar_bilateral = parametros.get("aplicar_bilateral", True)

    # --- AZUL: Total de Células (APLICANDO ALTA PRECISÃO) -----
    canal_azul = normalizar_robusto(b)
    param_azul_especifico = {
        "kernel_tophat": kernel_tophat,
        "aplicar_bilateral": aplicar_bilateral,
        "bias": parametros["bias_azul"],
        "min_circularidade": parametros["min_circularidade"],
        "area_maxima": parametros.get("area_maxima")
    }
    n_azul, lab_azul, mask_azul, thr_azul, p_azul = segmentar_canal(canal_azul, param_azul_especifico)

    # --- VERDE e AMARELO -----
    canal_verde = normalizar_robusto(g)
    corrigido_verde = corrigir_fundo(canal_verde, kernel_tophat, aplicar_bilateral)
    mask_verde, thr_verde, _ = limiar_otsu_com_ajuste(corrigido_verde, parametros["bias_verde"])
    mask_verde = limpar_mascara(mask_verde)

    # AJUSTE DE PRECISÃO: min(r,g) exige que os DOIS canais estejam altos ao
    # mesmo tempo, o que subestima marcadores de morte que tendem mais para
    # laranja/vermelho (ex: iodeto de propídeo) do que para um amarelo puro
    # equilibrado. A média (r+g)/2 é mais tolerante a esse desvio de matiz sem
    # deixar de descontar o azul de fundo.
    #
    # AJUSTE DE PRECISÃO 2: a média pura (r+g)/2 tem um efeito colateral sério
    # — um pixel VERDE PURO (célula viva, sem vermelho nenhum, ex: R=10,
    # G=200) ainda gera uma média de 105, alta o suficiente pra passar como
    # "amarelo" contra um fundo azul mais escuro. Isso faz o canal amarelo
    # "vazar" sinal de células vivas com verde forte, especialmente visível em
    # imagens com verde mais saturado (ex: BMP não-comprimido, sem o
    # achatamento de extremos que o JPEG aplica). Amarelo de verdade precisa
    # de vermelho E verde presentes; verde puro não é amarelo. Penalizamos
    # pixels onde o verde domina muito sobre o vermelho (sinal de "verde
    # puro", não de amarelo/laranja) proporcionalmente a esse desequilíbrio.
    excesso_verde = np.clip(g - r, 0, 255)
    amarelicidade = np.clip(((r + g) / 2.0) - b - excesso_verde * 0.5, 0, 255)
    canal_amarelo = normalizar_robusto(amarelicidade)
    corrigido_amarelo = corrigir_fundo(canal_amarelo, kernel_tophat, aplicar_bilateral)
    mask_amarelo, thr_amarelo, _ = limiar_otsu_com_ajuste(corrigido_amarelo, parametros["bias_amarelo"])
    mask_amarelo = limpar_mascara(mask_amarelo)

    # --- Classificação célula a célula -------
    raio_dilatacao = max(2, int(round(np.sqrt(p_azul.get("area_minima", 15) / np.pi) * 0.8)))
    classificacao = classificar_nucleos(
        lab_azul, mask_verde.astype(np.float32), mask_amarelo.astype(np.float32),
        raio_dilatacao=raio_dilatacao
    )

    n_verde = classificacao["n_viaveis"]
    n_amarelo = classificacao["n_mortas"]
    n_indeterminadas = classificacao["n_indeterminadas"]
    total_celulas = n_azul

    # --- AJUSTE DE PRECISÃO MATEMÁTICA NA FÓRMULA ---
    formula = parametros.get("formula_viabilidade", "Padrão Biológico (Vivas / (Vivas + Mortas))")
    if formula == "Padrão Biológico (Vivas / (Vivas + Mortas))":
        base_calculo = n_verde + n_amarelo
    else:
        base_calculo = total_celulas

    viabilidade = min((n_verde / base_calculo) * 100, 100.0) if base_calculo > 0 else 0.0

    eh_jpeg = nome_arquivo.lower().endswith((".jpg", ".jpeg"))

    saida = img.copy()
    saida = desenhar_contornos(saida, lab_azul, (255, 0, 0))                            
    saida = desenhar_contornos(saida, classificacao["labels_viaveis"], (0, 255, 0))      
    saida = desenhar_contornos(saida, classificacao["labels_mortas"], (0, 255, 255))     

    relatorio = {
        "Arquivo": nome_arquivo,
        "Total (Azul)": n_azul,
        "Viáveis (Verde)": n_verde,
        "Mortas (Amarelo)": n_amarelo,
        "Indeterminadas": n_indeterminadas,
        "Base de Cálculo": base_calculo,
        "Viabilidade (%)": round(viabilidade, 2),
        "Formato com perda (JPEG)": "Sim" if eh_jpeg else "Não",
    }

    texto_linhas = [
        f"Arquivo: {nome_arquivo}",
        f"Viabilidade: {relatorio['Viabilidade (%)']}%",
        f"Total: {relatorio['Base de Cálculo']}",
        f"Viaveis: {relatorio['Viáveis (Verde)']}",
        f"Mortas: {relatorio['Mortas (Amarelo)']}"
    ]

    y0, dy = 40, 40
    for i, linha in enumerate(texto_linhas):
        y = y0 + i * dy
        cv2.putText(saida, linha, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(saida, linha, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    saida_rgb = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)

    mascaras = {
        "azul": mask_azul,
        "verde": mask_verde,
        "amarelo": mask_amarelo,
        "limiares": {"azul": thr_azul, "verde": thr_verde, "amarelo": thr_amarelo},
        "params_auto": {"azul": p_azul, "raio_dilatacao_classificacao": raio_dilatacao},
        "gates": {"verde": thr_verde, "amarelo": thr_amarelo},
        "tabela_celulas": classificacao["tabela_celulas"],
        "eh_jpeg": eh_jpeg,
    }

    return relatorio, saida_rgb, saida, original_rgb, mascaras, None


# =========================================================================
# INTERFACE STREAMLIT
# =========================================================================
st.set_page_config(page_title="ViaCell", layout="wide")

st.title("🔬 ViaCell")
st.markdown("Faça o upload de múltiplas imagens de fluorescência para análise em lote.")

with st.sidebar:
    st.markdown("### ⚙️ Detecção Automática")
    
    aplicar_bilateral = st.checkbox(
        "Suavização Avançada (Filtro Bilateral)", value=True,
        help="Mantém as bordas das células cravadas, mas reduz falsos positivos gerados por ruído do fundo."
    )

    # --- AJUSTE DE PRECISÃO: Seleção de Fórmula de Cálculo ---
    formula_viabilidade = st.selectbox(
        "Fórmula de Viabilidade",
        options=[
            "Padrão Biológico (Vivas / (Vivas + Mortas))",
            "Rigoroso (Vivas / Total Azul)"
        ],
        index=0,
        help=(
            "Padrão Biológico: Despreza as células Indeterminadas (com sinal de fluorescência muito fraco). "
            "Rigoroso: Considera toda célula azul (mesmo sem sinal verde/amarelo) como não-viável."
        )
    )

    ajuste_fino = st.checkbox(
        "Ativar controles manuais de precisão", value=False
    )

    if ajuste_fino:
        st.markdown("**Limiares de Sensibilidade**")
        bias_azul = st.slider("Total (Azul)", -60, 60, 0, step=2)
        bias_verde = st.slider(
            "Viáveis (Verde)", -60, 60, 0, step=2,
            help="Bias positivo (+) diminui o limiar de corte, capturando mais células de fluorescência verde fraca."
        )
        bias_amarelo = st.slider("Mortas (Amarelo)", -60, 60, 0, step=2)
        
        st.markdown("**Filtros Geométricos (Exclusão de Ruídos)**")
        min_circularidade = st.slider(
            "Circularidade Mínima (0 a 1)", 0.0, 1.0, 0.2, step=0.05,
            help="Descarta detritos e sujeiras. 0 = aceita qualquer formato. 1 = apenas círculos perfeitos. 0.2 é um bom meio termo."
        )
        area_maxima = st.number_input(
            "Área Máxima por Célula (px)", value=10000, step=500,
            help="Descarta aglomerados de sujeira muito grandes que o watershed não separou."
        )
    else:
        bias_azul = bias_verde = bias_amarelo = 0
        # AJUSTE DE PRECISÃO: mesmo sem o usuário mexer em nada, aplica um piso
        # de circularidade que descarta ruído/detritos pontiagudos (specks e
        # fiapos), mas não é rígido a ponto de rejeitar núcleos levemente ovais.
        min_circularidade = 0.15
        area_maxima = None

    st.markdown("---")
    mostrar_mascaras = st.checkbox("Mostrar máscaras de calibração (debug)", value=False)

parametros = {
    "bias_azul": bias_azul,
    "bias_verde": bias_verde,
    "bias_amarelo": bias_amarelo,
    "kernel_tophat": None,
    "aplicar_bilateral": aplicar_bilateral,
    "min_circularidade": min_circularidade,
    "area_maxima": area_maxima,
    "formula_viabilidade": formula_viabilidade # <--- ADICIONADO
}

arquivos_upados = st.file_uploader(
    "Arraste as imagens ou clique para procurar",
    type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True
)

if "zip_gerado" not in st.session_state: st.session_state.zip_gerado = None
if "resultados" not in st.session_state: st.session_state.resultados = None
if "imagens_processadas" not in st.session_state: st.session_state.imagens_processadas = None

if arquivos_upados:
    if st.button("Processar Imagens", type="primary"):
        st.write("---")

        resultados = []
        imagens_para_salvar = []
        imagens_processadas = {}
        tabelas_por_imagem = {}
        tempos_processamento = []

        total_arquivos = len(arquivos_upados)
        barra_progresso = st.progress(0.0)
        texto_status = st.empty()
        galeria_ao_vivo = st.container()

        for idx, arquivo in enumerate(arquivos_upados):
            inicio = time.time()
            relatorio, img_rgb_tela, img_bgr_salvar, original_rgb, mascaras, erro = processar_imagem_na_memoria(
                arquivo.read(), arquivo.name, parametros
            )
            tempos_processamento.append(time.time() - inicio)

            if erro is not None:
                st.error(f"Erro ao processar '{arquivo.name}': {erro}")
            else:
                resultados.append(relatorio)
                imagens_para_salvar.append((arquivo.name, img_bgr_salvar))
                imagens_processadas[arquivo.name] = {
                    "relatorio": relatorio,
                    "processada_rgb": img_rgb_tela,
                    "original_rgb": original_rgb,
                    "mascaras": mascaras,
                }
                tabelas_por_imagem[arquivo.name] = mascaras.get("tabela_celulas", [])

                with galeria_ao_vivo:
                    with st.expander(f"✅ {arquivo.name} — Viabilidade: {relatorio['Viabilidade (%)']}%", expanded=False):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.write(f"**Total de Células:** {relatorio['Total (Azul)']}")
                            st.write(f"**Vivas:** {relatorio['Viáveis (Verde)']}")
                            st.write(f"**Mortas:** {relatorio['Mortas (Amarelo)']}")
                        with col_b:
                            st.image(img_rgb_tela, use_container_width=True)

            processados = idx + 1
            barra_progresso.progress(processados / total_arquivos)
            texto_status.info(f"Processadas {processados}/{total_arquivos} imagens...")

        texto_status.success(f"Concluído! {total_arquivos} imagens processadas.")
        
        st.session_state.resultados = resultados
        st.session_state.imagens_processadas = imagens_processadas

        if resultados:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                df_resultados_zip = pd.DataFrame(resultados)
                zip_file.writestr("relatorio_consolidado.csv", df_resultados_zip.to_csv(index=False).encode('utf-8'))

                for nome_original, tabela in tabelas_por_imagem.items():
                    if tabela:
                        nome_base = nome_original.rsplit('.', 1)[0]
                        df_celulas = pd.DataFrame(tabela)
                        zip_file.writestr(f"celulas_individuais_{nome_base}.csv", df_celulas.to_csv(index=False).encode('utf-8'))

                for nome_original, img_bgr in imagens_para_salvar:
                    nome_base = nome_original.rsplit('.', 1)[0]
                    sucesso, buffer_img = cv2.imencode('.png', img_bgr)
                    if sucesso:
                        zip_file.writestr(f"{nome_base}_resultado.png", buffer_img.tobytes())

            st.session_state.zip_gerado = zip_buffer.getvalue()

imagens_processadas = st.session_state.get("imagens_processadas")
resultados = st.session_state.get("resultados")

if imagens_processadas:
    st.write("---")
    st.markdown("#### Navegador de resultados")
    nomes_disponiveis = list(imagens_processadas.keys())

    with st.sidebar:
        st.markdown("### Resultados")
        imagem_selecionada = st.radio("Selecione uma imagem para visualizar", options=nomes_disponiveis, label_visibility="collapsed")

    dados_sel = imagens_processadas[imagem_selecionada]
    relatorio_sel = dados_sel["relatorio"]
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Resultados")
        st.metric("Viabilidade", f"{relatorio_sel['Viabilidade (%)']}%")
        st.write(f"**Total de Células:** {relatorio_sel['Total (Azul)']}")
        st.write(f"**Vivas:** {relatorio_sel['Viáveis (Verde)']}")
        st.write(f"**Mortas:** {relatorio_sel['Mortas (Amarelo)']}")
        
        modo_visualizacao = st.radio(
            "Visualizar",
            options=["Resultado processado", "Imagem original"] + (["Máscaras (debug)"] if mostrar_mascaras else []),
            horizontal=True,
        )

    with col2:
        if modo_visualizacao == "Imagem original":
            st.image(dados_sel["original_rgb"], caption=f"Imagem original - {imagem_selecionada}", use_container_width=True)
        elif modo_visualizacao == "Máscaras (debug)":
            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.image(dados_sel["mascaras"]["azul"], caption="Máscara Azul (Total)", use_container_width=True)
            with mc2: st.image(dados_sel["mascaras"]["verde"], caption="Máscara Verde (Viáveis)", use_container_width=True)
            with mc3: st.image(dados_sel["mascaras"]["amarelo"], caption="Máscara Amarela (Mortas)", use_container_width=True)
        else:
            st.image(dados_sel["processada_rgb"], caption=f"Contornos e Resultados - {imagem_selecionada}", use_container_width=True)

if resultados:
    st.write("---")
    st.subheader("📊 Relatório")
    st.dataframe(pd.DataFrame(resultados), use_container_width=True)

if st.session_state.get("zip_gerado"):
    st.download_button(
        label="📥 Baixar Todas as Imagens + Planilha (ZIP)",
        data=st.session_state.zip_gerado,
        file_name="ViaCell_Resultados.zip",
        mime="application/zip"
    )