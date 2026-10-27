
import cv2


# ==================================================
# 1. DETECÇÃO DOS CONTORNOS
# ==================================================

def detectar_contornos(grade):
    """
    Identifica os contornos presentes na imagem
    binária da grade.
    """

    contornos, _ = cv2.findContours(
        grade,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contornos


# ==================================================
# 2. SELEÇÃO DO MAIOR CONTORNO
# ==================================================

def selecionar_maior_contorno(contornos):
    """
    Seleciona o contorno com a maior área.

    Retorna None caso nenhum contorno seja encontrado.
    """

    if len(contornos) == 0:
        return None

    maior_contorno = max(
        contornos,
        key=cv2.contourArea
    )

    return maior_contorno


# ==================================================
# 3. DETECÇÃO DOS VÉRTICES
# ==================================================

def detectar_vertices(contorno):
    """
    Aproxima um contorno por um polígono
    e retorna os vértices encontrados.
    """

    perimetro = cv2.arcLength(
        contorno,
        True
    )

    aproximacao = cv2.approxPolyDP(
        contorno,
        0.02 * perimetro,
        True
    )

    return aproximacao
