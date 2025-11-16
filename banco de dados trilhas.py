import chromadb
import os
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from PIL import Image
import numpy as np
from chromadb.utils import embedding_functions
import base64
from io import BytesIO
import fitz  #PyMuPDF

#Configurações iniciais
pdf_folder = filedialog.askdirectory(
    title="Selecione a pasta com os PDFs (imagens)",
    initialdir=os.path.expanduser("~")
)
db_folder = os.path.join(os.path.dirname(__file__), "Banco de dados imagens trilhas")
COLLECTION_NAME = "Imagens_PDF_Collection"


def validar_ambiente():
    #Valida se pasta de PDFs existe e percorre todas as subpastas
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"❌ Pasta não encontrada: {pdf_folder}")

    pdfs = []
    pastas_encontradas = 0

    #Percorrer todas as subpastas recursivamente
    for root, dirs, files in os.walk(pdf_folder):
        pastas_encontradas += 1
        for f in files:
            if f.lower().endswith('.pdf'):
                caminho_completo = os.path.join(root, f)
                #Armazena (caminho_completo, caminho_relativo)
                caminho_relativo = os.path.relpath(caminho_completo, pdf_folder)
                pdfs.append((caminho_completo, caminho_relativo))

    if not pdfs:
        raise FileNotFoundError(f"❌ Nenhum PDF encontrado em: {pdf_folder} e subpastas")

    print(f"Encontrados {len(pdfs)} PDFs em {pastas_encontradas} pasta(s)\n")
    return pdfs


def extrair_imagens_pdf(caminho_pdf):
    #Extrai imagens do PDF usando múltiplas estratégias
    imagens_extraidas = []

    try:
        doc = fitz.open(caminho_pdf)
        print(f"     {len(doc)} página(s)")

        #ESTRATÉGIA 1: Tentar extrair imagens embutidas
        total_imagens_embutidas = 0

        for pagina_num in range(len(doc)):
            pagina = doc[pagina_num]
            lista_imagens = pagina.get_images(full=True)

            for img_index, img_info in enumerate(lista_imagens):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img = Image.open(BytesIO(image_bytes))

                    #Aceitar imagens de tamanho razoável
                    if img.size[0] >= 100 and img.size[1] >= 100:
                        info = {
                            'pagina': pagina_num + 1,
                            'indice_imagem': img_index + 1,
                            'dimensoes_originais': img.size,
                            'modo': img.mode,
                            'formato': image_ext,
                            'tamanho_bytes': len(image_bytes),
                            'metodo': 'embutida'
                        }

                        imagens_extraidas.append((img, info))
                        total_imagens_embutidas += 1

                except Exception as e:
                    continue

        if total_imagens_embutidas > 0:
            print(f"       {total_imagens_embutidas} imagem(ns) embutida(s) extraída(s)")

        #ESTRATÉGIA 2: SEMPRE renderizar páginas (para capturar mapas)
        print(f"      Renderizando páginas como imagens...")

        for pagina_num in range(len(doc)):
            pagina = doc[pagina_num]

            #Renderizar em alta resolução (300 DPI)
            matriz = fitz.Matrix(300 / 72, 300 / 72)
            pix = pagina.get_pixmap(matrix=matriz, alpha=False)

            #Converter para PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            info = {
                'pagina': pagina_num + 1,
                'indice_imagem': 1,
                'dimensoes_originais': img.size,
                'modo': img.mode,
                'formato': 'rendered_page',
                'tamanho_bytes': len(pix.samples),
                'metodo': 'renderizada',
                'dpi': 300
            }

            imagens_extraidas.append((img, info))
            print(f"      → Página {pagina_num + 1}: {img.size[0]}x{img.size[1]} pixels")

        doc.close()
        return imagens_extraidas

    except Exception as e:
        print(f"    ❌ Erro ao processar PDF: {e}")
        return []


def extrair_features_imagem(img):
    #Extrai features da imagem PIL e converte para embedding
    try:
        #Converter para RGB se necessário
        if img.mode != 'RGB':
            img = img.convert('RGB')

        #Redimensionar para tamanho padrão (224x224 é comum para embeddings)
        img_resized = img.resize((224, 224))

        #Converter para array numpy e normalizar
        img_array = np.array(img_resized).astype('float32') / 255.0

        #Criar embedding simples
        embedding = img_array.flatten().tolist()

        return embedding

    except Exception as e:
        print(f"    ❌ Erro ao processar imagem: {e}")
        return None


def imagem_para_base64(img, max_size=(800, 800)):
    #Converte imagem PIL para base64
    try:
        #Criar cópia para não modificar original
        img_copy = img.copy()

        #Redimensionar (se muito grande)
        img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)

        #Converter para base64
        buffered = BytesIO()
        img_copy.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return img_base64
    except:
        return None


def limpar_colecao_existente(collection):
    #Remove todos os documentos da coleção para evitar duplicatas
    try:
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"    Removidos {len(results['ids'])} itens antigos\n")
    except Exception as e:
        print(f"    Aviso ao limpar coleção: {e}\n")


def processar_pdfs():
    #Função principal que processa todos os PDFs e extrai imagens

    pdfs = validar_ambiente()

    #Criar pasta do banco se não existir
    os.makedirs(db_folder, exist_ok=True)

    #Inicializar ChromaDB com persistência
    print("Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=db_folder)

    #Criar coleção
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Coleção de imagens extraídas de PDFs"}
    )

    #Limpar dados antigos (opcional - comente se quiser manter)
    limpar_colecao_existente(collection)

    #Processar cada PDF
    total_imagens = 0
    pdfs_processados = 0
    pdfs_com_erro = 0
    inicio_geral = datetime.now()

    for idx, (caminho_pdf, caminho_relativo) in enumerate(pdfs, 1):
        nome_arquivo = os.path.basename(caminho_pdf)
        pasta_relativa = os.path.dirname(caminho_relativo)

        print(f"\n[{idx}/{len(pdfs)}]  {caminho_relativo}")
        if pasta_relativa:
            print(f"     Pasta: {pasta_relativa}")

        try:
            inicio = datetime.now()

            #Extrair imagens do PDF
            imagens = extrair_imagens_pdf(caminho_pdf)

            if not imagens:
                print(f"    ⚠️  Nenhuma imagem encontrada - pulando arquivo")
                pdfs_com_erro += 1
                continue

            #Processar cada imagem extraída
            imagens_processadas_pdf = 0

            for img, info in imagens:
                try:
                    #Extrair embedding
                    embedding = extrair_features_imagem(img)

                    if embedding is None:
                        continue

                    #Converter imagem para base64 (opcional, para preview)
                    img_base64 = imagem_para_base64(img)

                    #Metadados
                    metadata = {
                        "arquivo_pdf": nome_arquivo,
                        "caminho_relativo": caminho_relativo,
                        "pasta": pasta_relativa if pasta_relativa else "raiz",
                        "pagina": info['pagina'],
                        "indice_imagem": info['indice_imagem'],
                        "dimensoes": f"{info['dimensoes_originais'][0]}x{info['dimensoes_originais'][1]}",
                        "formato": info['formato'],
                        "tamanho_kb": round(info['tamanho_bytes'] / 1024, 2),
                        "data_processamento": datetime.now().isoformat(),
                    }

                    #Adicionar preview se disponível
                    if img_base64:
                        metadata["preview_base64"] = img_base64[:1000]  # Limitar tamanho

                    #Criar identificador único
                    doc_id = f"{caminho_relativo}_p{info['pagina']}_i{info['indice_imagem']}"

                    #Adicionar ao ChromaDB
                    collection.add(
                        ids=[str(uuid.uuid4())],
                        embeddings=[embedding],
                        documents=[doc_id],  #Identificador como documento
                        metadatas=[metadata]
                    )

                    imagens_processadas_pdf += 1
                    total_imagens += 1

                except Exception as e:
                    print(f"        ❌ Erro ao processar imagem: {e}")
                    continue

            tempo_decorrido = (datetime.now() - inicio).total_seconds()
            print(f"     {imagens_processadas_pdf} imagem(ns) processada(s) em {tempo_decorrido:.2f}s")

            pdfs_processados += 1

        except Exception as e:
            print(f"    ❌ Erro ao processar PDF: {e}")
            pdfs_com_erro += 1
            continue

    tempo_total = (datetime.now() - inicio_geral).total_seconds()

    print("\n" + "=" * 60)
    print(" RESUMO DO PROCESSAMENTO")
    print("=" * 60)
    print(f" PDFs processados com sucesso: {pdfs_processados}")
    print(f" PDFs com erro: {pdfs_com_erro}")
    print(f" Total de imagens extraídas: {total_imagens}")
    print(f" Tempo total: {tempo_total:.1f}s")
    print(f" Banco salvo em: {db_folder}")
    print(f" Coleção: {COLLECTION_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        processar_pdfs()
    except Exception as e:
        print(f"\n❌ ERRO NA EXECUÇÃO: {e}")
        import traceback

        traceback.print_exc()