
import cv2


# ==================================================
# 1. BINARIZAÇÃO DA IMAGEM
# ==================================================

def binarizar_imagem(imagem):
    """
    Converte uma imagem BGR em uma imagem binária
    com os elementos escuros destacados em branco.
    """

    # Conversão para escala de cinza
    imagem_cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    # Aplicação da binarização adaptativa
    imagem_adaptativa = cv2.adaptiveThreshold(
        imagem_cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    # Inversão: elementos escuros ficam brancos
    imagem_invertida = cv2.bitwise_not(
        imagem_adaptativa
    )

    return imagem_invertida


# ==================================================
# 2. EXTRAÇÃO DA GRADE
# ==================================================

def extrair_grade(imagem_binaria):
    """
    Identifica linhas horizontais e verticais
    utilizando operações morfológicas.
    """

    # Elemento estruturante horizontal
    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40, 1)
    )

    # Elemento estruturante vertical
    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 40)
    )

    # Detecção das linhas horizontais
    linhas_horizontais = cv2.morphologyEx(
        imagem_binaria,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    # Detecção das linhas verticais
    linhas_verticais = cv2.morphologyEx(
        imagem_binaria,
        cv2.MORPH_OPEN,
        kernel_vertical
    )

    # Combinação das linhas detectadas
    grade = cv2.bitwise_or(
        linhas_horizontais,
        linhas_verticais
    )

    return grade
