
import cv2
import matplotlib.pyplot as plt

from src.leitor_imagem import carregar_imagem
from src.perspectiva import corrigir_perspectiva


# ==================================================
# 1. CAMINHOS DAS IMAGENS
# ==================================================

caminhos = [
    "img/folha_teste_1 (1).jpg",
    "img/folha_teste_1 (2).jpg"
]


# ==================================================
# 2. PROCESSAMENTO DAS IMAGENS
# ==================================================

for caminho in caminhos:

    # ----------------------------------------------
    # 2.1. Carregamento da imagem
    # ----------------------------------------------

    imagem = carregar_imagem(caminho)

    altura, largura, canais = imagem.shape

    print(f"\nImagem: {caminho}")
    print("Imagem carregada com sucesso!")
    print(f"Largura: {largura} pixels")
    print(f"Altura: {altura} pixels")
    print(f"Canais de cor: {canais}")

    # ----------------------------------------------
    # 2.2. Conversão para tons de cinza
    # ----------------------------------------------

    imagem_cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------------------------
    # 2.3. Binarização adaptativa
    # ----------------------------------------------

    imagem_adaptativa = cv2.adaptiveThreshold(
        imagem_cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    # ----------------------------------------------
    # 2.4. Inversão da imagem
    # ----------------------------------------------

    imagem_invertida = cv2.bitwise_not(
        imagem_adaptativa
    )

    # ----------------------------------------------
    # 2.5. Elementos estruturantes
    # ----------------------------------------------

    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40, 1)
    )

    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 40)
    )

    # ----------------------------------------------
    # 2.6. Detecção das linhas horizontais
    # ----------------------------------------------

    linhas_horizontais = cv2.morphologyEx(
        imagem_invertida,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    # ----------------------------------------------
    # 2.7. Detecção das linhas verticais
    # ----------------------------------------------

    linhas_verticais = cv2.morphologyEx(
        imagem_invertida,
        cv2.MORPH_OPEN,
        kernel_vertical
    )

    # ----------------------------------------------
    # 2.8. Combinação das linhas
    # ----------------------------------------------

    grade = cv2.bitwise_or(
        linhas_horizontais,
        linhas_verticais
    )

    # ----------------------------------------------
    # 2.9. Detecção dos contornos
    # ----------------------------------------------

    contornos, _ = cv2.findContours(
        grade,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Cria uma cópia da imagem original
    imagem_contornos = imagem.copy()

    # ----------------------------------------------
    # 2.10. Seleção do maior contorno
    # ----------------------------------------------

    if len(contornos) > 0:

        maior_contorno = max(
            contornos,
            key=cv2.contourArea
        )

        area = cv2.contourArea(maior_contorno)

        print(f"Contornos encontrados: {len(contornos)}")
        print(f"Área do maior contorno: {area:.2f} px²")

        # Desenha o maior contorno em verde
        cv2.drawContours(
            imagem_contornos,
            [maior_contorno],
            -1,
            (0, 255, 0),
            3
        )

        # ------------------------------------------
        # 2.11. Cálculo do perímetro
        # ------------------------------------------

        perimetro = cv2.arcLength(
            maior_contorno,
            True
        )

        # ------------------------------------------
        # 2.12. Aproximação poligonal
        # ------------------------------------------

        aproximacao = cv2.approxPolyDP(
            maior_contorno,
            0.02 * perimetro,
            True
        )

        print(
            f"Vértices encontrados: {len(aproximacao)}"
        )

        # ------------------------------------------
        # 2.13. Desenho dos vértices
        # ------------------------------------------

        for ponto in aproximacao:

            x, y = ponto[0]

            cv2.circle(
                imagem_contornos,
                (int(x), int(y)),
                10,
                (0, 0, 255),
                -1
            )

        # ------------------------------------------
        # 2.14. Correção de perspectiva
        # ------------------------------------------

        if len(aproximacao) == 4:

            pontos = aproximacao.reshape(4, 2)

            imagem_corrigida = corrigir_perspectiva(
                imagem,
                pontos
            )

            # Converte a imagem corrigida para RGB
            imagem_corrigida_rgb = cv2.cvtColor(
                imagem_corrigida,
                cv2.COLOR_BGR2RGB
            )

            # Exibe a imagem corrigida
            plt.figure(figsize=(12, 8))

            plt.imshow(imagem_corrigida_rgb)

            plt.title(
                f"Perspectiva corrigida - {caminho}"
            )

            plt.axis("off")

        else:

            print(
                "Não foi possível corrigir a perspectiva: "
                f"{len(aproximacao)} vértices encontrados."
            )

    else:

        print("Nenhum contorno encontrado.")

    # ----------------------------------------------
    # 2.15. Conversão para RGB
    # ----------------------------------------------

    imagem_contornos_rgb = cv2.cvtColor(
        imagem_contornos,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------
    # 2.16. Visualização dos contornos
    # ----------------------------------------------

    plt.figure(figsize=(12, 8))

    plt.imshow(imagem_contornos_rgb)

    plt.title(
        f"Contorno e vértices detectados - {caminho}"
    )

    plt.axis("off")


# ==================================================
# 3. EXIBIÇÃO DAS IMAGENS
# ==================================================

plt.show()
