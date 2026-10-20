
import cv2
import numpy as np


def corrigir_perspectiva(imagem, pontos):

    # Converte os pontos para uma matriz
    pontos = np.array(pontos, dtype="float32")

    # Ordena os quatro cantos
    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1)

    ordenados = np.zeros((4, 2), dtype="float32")

    ordenados[0] = pontos[np.argmin(soma)]
    ordenados[2] = pontos[np.argmax(soma)]
    ordenados[1] = pontos[np.argmin(diferenca)]
    ordenados[3] = pontos[np.argmax(diferenca)]

    superior_esquerdo, superior_direito, inferior_direito, inferior_esquerdo = ordenados

    # Calcula a largura
    largura_inferior = np.linalg.norm(
        inferior_direito - inferior_esquerdo
    )

    largura_superior = np.linalg.norm(
        superior_direito - superior_esquerdo
    )

    largura = int(max(largura_inferior, largura_superior))

    # Calcula a altura
    altura_direita = np.linalg.norm(
        superior_direito - inferior_direito
    )

    altura_esquerda = np.linalg.norm(
        superior_esquerdo - inferior_esquerdo
    )

    altura = int(max(altura_direita, altura_esquerda))

    if largura < 2 or altura < 2:
        raise ValueError("Dimensões inválidas para correção de perspectiva.")

    # Define os pontos de destino
    destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    # Calcula a matriz de transformação
    matriz = cv2.getPerspectiveTransform(
        ordenados,
        destino
    )

    # Aplica a transformação
    imagem_corrigida = cv2.warpPerspective(
        imagem,
        matriz,
        (largura, altura)
    )

    return imagem_corrigida
