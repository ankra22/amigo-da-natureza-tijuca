import chromadb
import pdfplumber
import os
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from chromadb.utils import embedding_functions

pdf_folder = filedialog.askdirectory(title="Selecione a pasta com os arquivos PDF",initialdir=os.path.expanduser("~"))
db_folder = os.path.join(os.path.dirname(__file__), "Banco de dados")
COLLECTION_NAME = "PlanoManejo_Tijuca"
CHUNK_SIZE = 1000
OVERLAP = 200

def validar_ambiente():
    #Valida se pasta de PDFs existe e contém arquivos
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f" Pasta não encontrada: {pdf_folder}")

    pdfs = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if not pdfs:
        raise FileNotFoundError(f" Nenhum PDF encontrado em: {pdf_folder}")

    print(f" Encontrados {len(pdfs)} PDFs para processar\n")
    return pdfs

def extrair_texto_pdf(caminho_pdf):
    #Extrai texto do PDF com tratamento de erros
    texto = ""
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            print(f"    Processando {len(pdf.pages)} páginas...")
            for i, page in enumerate(pdf.pages, 1):
                try:
                    t = page.extract_text()
                    if t:
                        texto += t + "\n"

                    #Mostra progresso a cada 10 páginas
                    if i % 10 == 0:
                        print(f"      → Página {i}/{len(pdf.pages)}")

                except Exception as e:
                    print(f"        Erro na página {i}: {e}")
                    continue

    except Exception as e:
        print(f"    Erro ao abrir PDF: {e}")
        return ""

    return texto

def criar_chunks_com_overlap(texto, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    #Divide texto em chunks com overlap, respeitando limites de palavras
    if not texto or not texto.strip():
        return []

    #Remove espaços múltiplos e linhas vazias excessivas
    texto = ' '.join(texto.split())

    chunks = []
    inicio = 0

    while inicio < len(texto):
        #Define fim do chunk
        fim = inicio + chunk_size

        #Se não é o último chunk, tenta terminar em espaço
        if fim < len(texto):
            #Procura último espaço antes do limite
            ultimo_espaco = texto.rfind(' ', inicio, fim)
            if ultimo_espaco > inicio:
                fim = ultimo_espaco

        chunk = texto[inicio:fim].strip()

        if chunk:  #Só adiciona se não for vazio
            chunks.append(chunk)

        #Move para próximo chunk com overlap
        inicio = fim - overlap if fim < len(texto) else fim

    return chunks


def limpar_colecao_existente(collection):
    #Remove todos os documentos da coleção para evitar duplicatas
    try:
        #Pega todos os IDs
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"  Removidos {len(results['ids'])} documentos antigos\n")
    except Exception as e:
        print(f"  Aviso ao limpar coleção: {e}\n")


def processar_pdfs():
    #Função principal que processa todos os PDFs

    #Validar ambiente
    pdfs = validar_ambiente()

    #Criar pasta do banco se não existir
    os.makedirs(db_folder, exist_ok=True)

    #emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    #Inicializar ChromaDB com persistência
    print(" Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=db_folder)
    collection = client.get_or_create_collection(name=COLLECTION_NAME) #embedding_function=emb_fn)
    #Limpar dados antigos (opcional - comente se quiser manter)
    limpar_colecao_existente(collection)

    #Processar cada PDF
    total_chunks = 0
    pdfs_processados = 0
    pdfs_com_erro = 0

    inicio_geral = datetime.now()

    for idx, nome_arquivo in enumerate(pdfs, 1):
        print(f"\n[{idx}/{len(pdfs)}]  {nome_arquivo}")
        caminho_pdf = os.path.join(pdf_folder, nome_arquivo)

        try:
            #Extrair texto
            inicio = datetime.now()
            texto = extrair_texto_pdf(caminho_pdf)

            if not texto.strip():
                print(f"     Nenhum texto extraído - pulando arquivo")
                pdfs_com_erro += 1
                continue

            #Criar chunks com overlap
            chunks = criar_chunks_com_overlap(texto)

            if not chunks:
                print(f"     Nenhum chunk criado - pulando arquivo")
                pdfs_com_erro += 1
                continue

            #Adicionar ao ChromaDB
            collection.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=chunks,
                metadatas=[{
                    "arquivo": nome_arquivo,
                    "parte": i + 1,
                    "total_partes": len(chunks),
                    "tamanho_original": len(texto),
                    "data_processamento": datetime.now().isoformat()
                } for i in range(len(chunks))]
            )

            tempo_decorrido = (datetime.now() - inicio).total_seconds()
            print(f"    {len(chunks)} chunks criados em {tempo_decorrido:.1f}s")

            total_chunks += len(chunks)
            pdfs_processados += 1

        except Exception as e:
            print(f"    Erro ao processar: {e}")
            pdfs_com_erro += 1
            continue

    #Resumo final
    tempo_total = (datetime.now() - inicio_geral).total_seconds()

    print("\n RESUMO DO PROCESSAMENTO")
    print(f" PDFs processados com sucesso: {pdfs_processados}")
    print(f" PDFs com erro: {pdfs_com_erro}")
    print(f" Total de chunks criados: {total_chunks}")
    print(f" Tempo total: {tempo_total:.1f}s")
    print(f" Banco salvo em: {db_folder}")

if __name__ == "__main__":
    try:
        processar_pdfs()
    except Exception as e:
        print(f"\n ERRO NA LEITURA: {e}")
        import traceback

        traceback.print_exc()