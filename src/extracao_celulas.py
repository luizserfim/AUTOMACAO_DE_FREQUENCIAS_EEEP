
import cv2
import numpy as np


# ==================================================
# 1. INTERPOLAÇÃO DAS LINHAS
# ==================================================

def obter_y(linha, x):
    """
    Calcula a coordenada Y de uma linha horizontal
    na posição X informada.
    """

    pontos = np.asarray(linha, dtype=np.float64)
    pontos = pontos[np.argsort(pontos[:, 0])]

    xs, indices = np.unique(
        pontos[:, 0],
        return_index=True
    )
    ys = pontos[indices, 1]

    if len(xs) < 2:
        return float(ys[0])

    return float(np.interp(x, xs, ys))


def obter_x(linha, y):
    """
    Calcula a coordenada X de uma linha vertical
    na posição Y informada.
    """

    pontos = np.asarray(linha, dtype=np.float64)
    pontos = pontos[np.argsort(pontos[:, 1])]

    ys, indices = np.unique(
        pontos[:, 1],
        return_index=True
    )
    xs = pontos[indices, 0]

    if len(ys) < 2:
        return float(xs[0])

    return float(np.interp(y, ys, xs))


# ==================================================
# 2. CÁLCULO DAS INTERSEÇÕES
# ==================================================

def calcular_intersecao(
    linha_horizontal,
    linha_vertical,
    iteracoes=8
):
    """
    Calcula aproximadamente o encontro entre
    uma linha horizontal e uma vertical.
    """

    pontos_h = np.asarray(
        linha_horizontal,
        dtype=np.float64
    )

    pontos_v = np.asarray(
        linha_vertical,
        dtype=np.float64
    )

    x = float(np.mean(pontos_v[:, 0]))
    y = float(np.mean(pontos_h[:, 1]))

    for _ in range(iteracoes):
        y = obter_y(linha_horizontal, x)
        x = obter_x(linha_vertical, y)

    return (x, y)


# ==================================================
# 3. EXTRAÇÃO DAS CÉLULAS
# ==================================================

def extrair_celulas(
    imagem_corrigida,
    linhas_horizontais,
    linhas_verticais,
    margem=2
):
    """
    Extrai células utilizando as interseções
    entre as linhas reconstruídas.

    Retorna uma lista de dicionários contendo:
        linha
        coluna
        imagem
        pontos
    """

    altura, largura = imagem_corrigida.shape[:2]

    horizontais = sorted(
        linhas_horizontais,
        key=lambda linha: np.mean(
            [p[1] for p in linha]
        )
    )

    verticais = sorted(
        linhas_verticais,
        key=lambda linha: np.mean(
            [p[0] for p in linha]
        )
    )

    celulas = []

    if len(horizontais) < 2 or len(verticais) < 2:
        return celulas

    # Matriz de interseções
    intersecoes = []

    for linha_h in horizontais:

        pontos_linha = []

        for linha_v in verticais:

            ponto = calcular_intersecao(
                linha_h,
                linha_v
            )

            pontos_linha.append(ponto)

        intersecoes.append(pontos_linha)

    # Percorre os espaços entre linhas consecutivas
    for i in range(len(horizontais) - 1):

        for j in range(len(verticais) - 1):

            superior_esquerdo = intersecoes[i][j]
            superior_direito = intersecoes[i][j + 1]
            inferior_direito = intersecoes[i + 1][j + 1]
            inferior_esquerdo = intersecoes[i + 1][j]

            origem = np.float32([
                superior_esquerdo,
                superior_direito,
                inferior_direito,
                inferior_esquerdo
            ])

            largura_superior = np.linalg.norm(
                origem[1] - origem[0]
            )
            largura_inferior = np.linalg.norm(
                origem[2] - origem[3]
            )

            altura_esquerda = np.linalg.norm(
                origem[3] - origem[0]
            )
            altura_direita = np.linalg.norm(
                origem[2] - origem[1]
            )

            largura_celula = int(round(
                max(largura_superior, largura_inferior)
            ))

            altura_celula = int(round(
                max(altura_esquerda, altura_direita)
            ))

            # Ignora regiões degeneradas
            if largura_celula <= 2 * margem + 2:
                continue

            if altura_celula <= 2 * margem + 2:
                continue

            # Ignora pontos claramente fora da imagem
            if (
                np.any(origem[:, 0] < 0)
                or np.any(origem[:, 0] >= largura)
                or np.any(origem[:, 1] < 0)
                or np.any(origem[:, 1] >= altura)
            ):
                continue

            destino = np.float32([
                [0, 0],
                [largura_celula - 1, 0],
                [largura_celula - 1, altura_celula - 1],
                [0, altura_celula - 1]
            ])

            matriz = cv2.getPerspectiveTransform(
                origem,
                destino
            )

            recorte = cv2.warpPerspective(
                imagem_corrigida,
                matriz,
                (largura_celula, altura_celula)
            )

            # Remove parte das bordas da célula
            recorte = recorte[
                margem:altura_celula - margem,
                margem:largura_celula - margem
            ]

            celulas.append({
                "linha": i,
                "coluna": j,
                "imagem": recorte,
                "pontos": origem
            })

    return celulas
