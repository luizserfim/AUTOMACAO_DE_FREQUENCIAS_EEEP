
import cv2
import numpy as np


# ==================================================
# 1. PRÉ-PROCESSAMENTO DAS CÉLULAS
# ==================================================

def preprocessar_celula(imagem_celula, margem=2):
    """
    Prepara uma célula para análise das marcações.

    Etapas:
        1. Conversão para tons de cinza
        2. Remoção de bordas
        3. Suavização
        4. Binarização

    Retorna:
        Imagem binária com as marcações em branco
        e o fundo em preto.
    """

    # ----------------------------------------------
    # 1.1. Conversão para escala de cinza
    # ----------------------------------------------

    if imagem_celula.ndim == 3:

        imagem_cinza = cv2.cvtColor(
            imagem_celula,
            cv2.COLOR_BGR2GRAY
        )

    else:
        imagem_cinza = imagem_celula.copy()

    # ----------------------------------------------
    # 1.2. Remoção de bordas
    # ----------------------------------------------

    altura, largura = imagem_cinza.shape

    if altura > 2 * margem and largura > 2 * margem:

        imagem_cinza = imagem_cinza[
            margem:altura - margem,
            margem:largura - margem
        ]

    # ----------------------------------------------
    # 1.3. Suavização da imagem
    # ----------------------------------------------

    imagem_suavizada = cv2.GaussianBlur(
        imagem_cinza,
        (3, 3),
        0
    )

    # ----------------------------------------------
    # 1.4. Binarização pelo método de Otsu
    # ----------------------------------------------

    _, imagem_binaria = cv2.threshold(
        imagem_suavizada,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return imagem_binaria

def analisar_componentes(imagem_binaria, area_minima=3):
    """
    Analisa os componentes brancos de uma célula binarizada.

    Parâmetros:
        imagem_binaria: imagem com marcações brancas
        area_minima: área mínima de um componente, em pixels

    Retorna:
        Um dicionário com as medidas da célula.
    """

    altura, largura = imagem_binaria.shape[:2]

    area_total = altura * largura

    # Quantidade total de pixels brancos
    pixels_brancos = cv2.countNonZero(imagem_binaria)

    # Percentual ocupado por pixels brancos
    percentual_ocupacao = (
        100 * pixels_brancos / area_total
        if area_total > 0 else 0
    )

    # Identificação dos componentes conectados
    quantidade_rotulos, rotulos, estatisticas, centroides = (
        cv2.connectedComponentsWithStats(
            imagem_binaria,
            connectivity=8
        )
    )

    componentes = []

    # O rótulo 0 representa o fundo
    for indice in range(1, quantidade_rotulos):

        x = int(estatisticas[indice, cv2.CC_STAT_LEFT])
        y = int(estatisticas[indice, cv2.CC_STAT_TOP])

        largura_componente = int(
            estatisticas[indice, cv2.CC_STAT_WIDTH]
        )

        altura_componente = int(
            estatisticas[indice, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            estatisticas[indice, cv2.CC_STAT_AREA]
        )

        if area < area_minima:
            continue

        componentes.append({
            "x": x,
            "y": y,
            "largura": largura_componente,
            "altura": altura_componente,
            "area": area
        })

    return {
        "largura_celula": largura,
        "altura_celula": altura,
        "pixels_brancos": pixels_brancos,
        "percentual_ocupacao": percentual_ocupacao,
        "quantidade_componentes": len(componentes),
        "componentes": componentes
    }