
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.leitor_imagem import carregar_imagem
from src.processamento import binarizar_imagem, extrair_grade
from src.detector_grade import (
    detectar_contornos,
    selecionar_maior_contorno,
    detectar_vertices
)
from src.segmentacao import (
    detectar_linhas_internas,
    localizar_linhas,
    localizar_linhas_por_faixas,
    reconstruir_linhas_horizontais,
    localizar_verticais_por_faixas,
    reconstruir_linhas_verticais
)
from src.perspectiva import corrigir_perspectiva
from src.extracao_celulas import extrair_celulas


# ==================================================
# 1. CONFIGURAÇÕES
# ==================================================

caminhos = [
    "img/folha_teste_1 (1).jpg",
    "img/folha_teste_1 (2).jpg"
]

# Linha que será utilizada para visualizar os recortes.
# 0 = cabeçalho
# 1 = primeira linha de estudantes
# 2 = segunda linha de estudantes, e assim por diante.
LINHA_ALUNO = 1

QUANTIDADE_CELULAS = 12


# ==================================================
# 2. FUNÇÃO AUXILIAR DE VISUALIZAÇÃO
# ==================================================

def mostrar_imagem(imagem, titulo, escala_cinza=False):
    """
    Exibe uma imagem utilizando Matplotlib.
    """

    plt.figure(figsize=(12, 8))

    if escala_cinza:
        plt.imshow(imagem, cmap="gray")
    else:
        imagem_rgb = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )
        plt.imshow(imagem_rgb)

    plt.title(titulo)
    plt.axis("off")


# ==================================================
# 3. VISUALIZAÇÃO DAS CÉLULAS DOS ESTUDANTES
# ==================================================

def mostrar_celulas(
    celulas,
    caminho,
    linha_aluno=1,
    quantidade=12
):
    """
    Exibe células de uma linha específica da tabela.

    O cabeçalho corresponde à linha 0.
    """

    celulas_filtradas = [
        celula
        for celula in celulas
        if celula["linha"] == linha_aluno
    ]

    celulas_filtradas.sort(
        key=lambda celula: celula["coluna"]
    )

    amostras = celulas_filtradas[:quantidade]

    if not amostras:
        print(
            f"Nenhuma célula encontrada "
            f"para a linha {linha_aluno}."
        )
        return

    fig, eixos = plt.subplots(
        3,
        4,
        figsize=(12, 6)
    )

    for eixo in eixos.flat:
        eixo.axis("off")

    for eixo, celula in zip(eixos.flat, amostras):

        imagem_rgb = cv2.cvtColor(
            celula["imagem"],
            cv2.COLOR_BGR2RGB
        )

        eixo.imshow(imagem_rgb)

        eixo.set_title(
            f"Linha {celula['linha']} | "
            f"Coluna {celula['coluna']}"
        )

    fig.suptitle(
        f"CÉLULAS DO ESTUDANTE - {caminho}"
    )

    plt.tight_layout()


# ==================================================
# 4. PROCESSAMENTO DAS IMAGENS
# ==================================================

for caminho in caminhos:

    print("\n" + "=" * 55)
    print(f"PROCESSANDO: {caminho}")
    print("=" * 55)

    # ----------------------------------------------
    # 4.1. Carregamento da imagem
    # ----------------------------------------------

    imagem = carregar_imagem(caminho)

    altura, largura, canais = imagem.shape

    print("Imagem carregada com sucesso!")
    print(f"Largura: {largura} pixels")
    print(f"Altura: {altura} pixels")
    print(f"Canais de cor: {canais}")

    # ----------------------------------------------
    # 4.2. Binarização e extração da grade
    # ----------------------------------------------

    imagem_binaria = binarizar_imagem(imagem)

    grade = extrair_grade(imagem_binaria)

    # ----------------------------------------------
    # 4.3. Detecção dos contornos
    # ----------------------------------------------

    contornos = detectar_contornos(grade)

    maior_contorno = selecionar_maior_contorno(
        contornos
    )

    if maior_contorno is None:
        print("Nenhum contorno encontrado.")
        continue

    area = cv2.contourArea(maior_contorno)

    print(f"Contornos encontrados: {len(contornos)}")
    print(f"Área do maior contorno: {area:.2f} px²")

    # ----------------------------------------------
    # 4.4. Detecção dos vértices
    # ----------------------------------------------

    aproximacao = detectar_vertices(
        maior_contorno
    )

    print(f"Vértices encontrados: {len(aproximacao)}")

    imagem_contornos = imagem.copy()

    cv2.drawContours(
        imagem_contornos,
        [maior_contorno],
        -1,
        (0, 255, 0),
        3
    )

    for ponto in aproximacao:

        x, y = ponto[0]

        cv2.circle(
            imagem_contornos,
            (int(x), int(y)),
            10,
            (0, 0, 255),
            -1
        )

    mostrar_imagem(
        imagem_contornos,
        f"Contorno e vértices detectados - {caminho}"
    )

    # ----------------------------------------------
    # 4.5. Verificação dos quatro vértices
    # ----------------------------------------------

    if len(aproximacao) != 4:

        print(
            "Não foi possível corrigir a perspectiva: "
            f"{len(aproximacao)} vértices encontrados."
        )

        continue

    # ----------------------------------------------
    # 4.6. Correção de perspectiva
    # ----------------------------------------------

    pontos = aproximacao.reshape(4, 2)

    imagem_corrigida = corrigir_perspectiva(
        imagem,
        pontos
    )

    altura_corrigida, largura_corrigida = (
        imagem_corrigida.shape[:2]
    )

    mostrar_imagem(
        imagem_corrigida,
        f"Perspectiva corrigida - {caminho}"
    )

    # ----------------------------------------------
    # 4.7. Detecção das linhas internas
    # ----------------------------------------------

    (
        linhas_horizontais,
        linhas_verticais,
        grade_interna
    ) = detectar_linhas_internas(
        imagem_corrigida
    )

    mostrar_imagem(
        grade_interna,
        f"Linhas internas detectadas - {caminho}",
        escala_cinza=True
    )

    # ----------------------------------------------
    # 4.8. Localização global das linhas
    # ----------------------------------------------

    (
        posicoes_horizontais,
        posicoes_verticais
    ) = localizar_linhas(
        linhas_horizontais,
        linhas_verticais
    )

    print("\nLOCALIZAÇÃO GLOBAL")
    print("Linhas horizontais:", len(posicoes_horizontais))
    print("Linhas verticais:", len(posicoes_verticais))

    # ----------------------------------------------
    # 4.9. Localização horizontal por faixas
    # ----------------------------------------------

    resultados_horizontais = localizar_linhas_por_faixas(
        linhas_horizontais,
        quantidade_faixas=8,
        percentual_minimo=0.45
    )

    print("\nDETECÇÃO HORIZONTAL POR FAIXAS")

    for resultado in resultados_horizontais:

        print(
            f"Faixa {resultado['faixa']}: "
            f"{len(resultado['posicoes'])} linhas"
        )

    # ----------------------------------------------
    # 4.10. Reconstrução horizontal
    # ----------------------------------------------

    linhas_horizontais_reconstruidas = (
        reconstruir_linhas_horizontais(
            resultados_horizontais,
            distancia_maxima=8,
            minimo_faixas=3,
            max_faixas_ausentes=1,
            largura_imagem=largura_corrigida,
            altura_imagem=altura_corrigida
        )
    )

    print(
        "\nLinhas horizontais reconstruídas:",
        len(linhas_horizontais_reconstruidas)
    )

    # ----------------------------------------------
    # 4.11. Localização vertical por faixas
    # ----------------------------------------------

    resultados_verticais = localizar_verticais_por_faixas(
        linhas_verticais,
        quantidade_faixas=8,
        percentual_minimo=0.45,
        distancia_agrupamento=3
    )

    print("\nDETECÇÃO VERTICAL POR FAIXAS")

    for resultado in resultados_verticais:

        print(
            f"Faixa {resultado['faixa']}: "
            f"{len(resultado['posicoes'])} linhas"
        )

    # ----------------------------------------------
    # 4.12. Reconstrução vertical
    # ----------------------------------------------

    linhas_verticais_reconstruidas = (
        reconstruir_linhas_verticais(
            resultados_verticais,
            distancia_maxima=5,
            minimo_faixas=3,
            max_faixas_ausentes=1,
            altura_imagem=altura_corrigida,
            largura_imagem=largura_corrigida
        )
    )

    print(
        "\nLinhas verticais reconstruídas:",
        len(linhas_verticais_reconstruidas)
    )

    # ----------------------------------------------
    # 4.13. Visualização horizontal por faixas
    # ----------------------------------------------

    imagem_faixas_horizontais = imagem_corrigida.copy()

    for resultado in resultados_horizontais:

        x_inicio = resultado["x_inicio"]
        x_fim = resultado["x_fim"]

        for y in resultado["posicoes"]:

            cv2.line(
                imagem_faixas_horizontais,
                (x_inicio, int(y)),
                (x_fim - 1, int(y)),
                (0, 255, 0),
                2
            )

        cv2.line(
            imagem_faixas_horizontais,
            (x_inicio, 0),
            (x_inicio, altura_corrigida - 1),
            (255, 0, 0),
            1
        )

    mostrar_imagem(
        imagem_faixas_horizontais,
        f"Detecção horizontal por faixas - {caminho}"
    )

    # ----------------------------------------------
    # 4.14. Visualização vertical por faixas
    # ----------------------------------------------

    imagem_faixas_verticais = imagem_corrigida.copy()

    for resultado in resultados_verticais:

        y_inicio = resultado["y_inicio"]
        y_fim = resultado["y_fim"]

        for x in resultado["posicoes"]:

            cv2.line(
                imagem_faixas_verticais,
                (int(x), y_inicio),
                (int(x), y_fim - 1),
                (0, 0, 255),
                2
            )

        cv2.line(
            imagem_faixas_verticais,
            (0, y_inicio),
            (largura_corrigida - 1, y_inicio),
            (255, 0, 0),
            1
        )

    mostrar_imagem(
        imagem_faixas_verticais,
        f"Detecção vertical por faixas - {caminho}"
    )

    # ----------------------------------------------
    # 4.15. Visualização global das posições
    # ----------------------------------------------

    imagem_global = imagem_corrigida.copy()

    for y in posicoes_horizontais:

        cv2.line(
            imagem_global,
            (0, int(y)),
            (largura_corrigida - 1, int(y)),
            (0, 255, 0),
            2
        )

    for x in posicoes_verticais:

        cv2.line(
            imagem_global,
            (int(x), 0),
            (int(x), altura_corrigida - 1),
            (0, 0, 255),
            2
        )

    mostrar_imagem(
        imagem_global,
        f"Posições globais das linhas - {caminho}"
    )

    # ----------------------------------------------
    # 4.16. Visualização horizontal reconstruída
    # ----------------------------------------------

    imagem_horizontal = imagem_corrigida.copy()

    for pontos_linha in linhas_horizontais_reconstruidas:

        pontos_np = np.array(
            pontos_linha,
            dtype=np.int32
        )

        cv2.polylines(
            imagem_horizontal,
            [pontos_np],
            False,
            (0, 255, 0),
            2
        )

    mostrar_imagem(
        imagem_horizontal,
        f"Linhas horizontais reconstruídas - {caminho}"
    )

    # ----------------------------------------------
    # 4.17. Visualização vertical reconstruída
    # ----------------------------------------------

    imagem_vertical = imagem_corrigida.copy()

    for pontos_linha in linhas_verticais_reconstruidas:

        pontos_np = np.array(
            pontos_linha,
            dtype=np.int32
        )

        cv2.polylines(
            imagem_vertical,
            [pontos_np],
            False,
            (0, 0, 255),
            2
        )

    mostrar_imagem(
        imagem_vertical,
        f"Linhas verticais reconstruídas - {caminho}"
    )

    # ----------------------------------------------
    # 4.18. Reconstrução da grade completa
    # ----------------------------------------------

    imagem_grade_completa = imagem_corrigida.copy()

    for pontos_linha in linhas_horizontais_reconstruidas:

        pontos_np = np.array(
            pontos_linha,
            dtype=np.int32
        )

        cv2.polylines(
            imagem_grade_completa,
            [pontos_np],
            False,
            (0, 255, 0),
            2
        )

    for pontos_linha in linhas_verticais_reconstruidas:

        pontos_np = np.array(
            pontos_linha,
            dtype=np.int32
        )

        cv2.polylines(
            imagem_grade_completa,
            [pontos_np],
            False,
            (0, 0, 255),
            2
        )

    mostrar_imagem(
        imagem_grade_completa,
        f"GRADE COMPLETA RECONSTRUÍDA - {caminho}"
    )

    # ----------------------------------------------
    # 4.19. Extração individual das células
    # ----------------------------------------------

    celulas = extrair_celulas(
        imagem_corrigida,
        linhas_horizontais_reconstruidas,
        linhas_verticais_reconstruidas,
        margem=2
    )

    print("\nEXTRAÇÃO DAS CÉLULAS")
    print(f"Total de células extraídas: {len(celulas)}")

    # ----------------------------------------------
    # 4.20. Visualização das células dos estudantes
    # ----------------------------------------------

    mostrar_celulas(
        celulas,
        caminho,
        linha_aluno=LINHA_ALUNO,
        quantidade=QUANTIDADE_CELULAS
    )

    # ----------------------------------------------
    # 4.21. Resumo do processamento
    # ----------------------------------------------

    print("\nRESUMO DO PROCESSAMENTO")

    print(
        "Linhas horizontais reconstruídas:",
        len(linhas_horizontais_reconstruidas)
    )

    print(
        "Linhas verticais reconstruídas:",
        len(linhas_verticais_reconstruidas)
    )

    print(
        "Células extraídas:",
        len(celulas)
    )

    print("Processamento concluído!")


# ==================================================
# 5. EXIBIÇÃO DAS IMAGENS
# ==================================================

plt.show()
