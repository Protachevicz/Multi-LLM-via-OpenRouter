import hashlib, random, math
from datetime import datetime

# Banco vetorizado simulado (lista de dicts com embedding e dados)
vector_db = []

def vectorizar_texto(texto, dim=64):
    """Gera embedding simulado determinístico para um texto."""
    # Usamos hash do texto como seed para gerar vetor pseudoaleatório fixo
    h = int(hashlib.sha256(texto.encode('utf-8')).hexdigest(), 16) 
    random.seed(h)
    # Vetor de dimensão fixa 'dim' com valores float no intervalo [0,1)
    return [random.random() for _ in range(dim)]

def router_modelo(pergunta):
    """Decide qual modelo LLM usar (heurística simples baseada no tamanho da pergunta)."""
    palavras = pergunta.split()
    if len(palavras) < 10:
        return "openai/gpt-3.5-turbo"   # modelo mais barato/rápido
    else:
        return "openai/gpt-4"          # modelo mais avançado (caro, performático)

def chamar_modelo(modelo, pergunta):
    """Simula chamada ao modelo selecionado via OpenRouter e retorna resposta."""
    # Em cenário real, aqui usaríamos a API do OpenRouter:
    # ex: openrouter.complete(prompt=..., model=modelo)
    return f"[Simulação de resposta do modelo {modelo} à pergunta: '{pergunta}']"

def armazenar_interacao(pergunta, resposta, modelo):
    """Armazena pergunta & resposta no banco vetorizado com seu embedding."""
    embedding = vectorizar_texto(pergunta)
    log_entry = {
        "pergunta": pergunta,
        "resposta": resposta,
        "modelo": modelo,
        "embedding": embedding,
        "timestamp": datetime.now()
    }
    vector_db.append(log_entry)

def cosine(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def buscar_similar(pergunta, threshold=0.8):
    """Busca pergunta similar no banco vetorizado via similaridade de cosseno."""
    if not vector_db:
        return None
    query_emb = vectorizar_texto(pergunta)
    best_match = None
    best_score = 0.0
    for entry in vector_db:
        score = cosine(query_emb, entry["embedding"])
        if score > best_score:
            best_score = score
            best_match = entry
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None

import time  # Adicione no topo do seu script, junto com os outros imports

if __name__ == "__main__":
    perguntas = [
        "Qual o prazo de entrega do produto?",
        "Como faço para cancelar minha assinatura?",
        "Existe parcelamento sem juros?",
        "Quais modelos estão disponíveis?",
        "Como atualizar meus dados cadastrais?",
        "Quais formas de pagamento aceitam?",
        "Como alterar o endereço de entrega?",
        "Como faço para trocar um produto com defeito?",
        "Quais modelos estão disponíveis?",
        "Qual o horário de atendimento?",
        "Como recebo a segunda via do boleto?",
        "Quais são os benefícios do clube de vantagens?",
        "Como faço para cancelar minha assinatura?",  # Pergunta repetida
        "Existe parcelamento sem juros?",              # Pergunta repetida
        "Qual o prazo de entrega do produto?",         # Pergunta repetida
        "Como atualizar meus dados cadastrais?",       # Pergunta repetida
        "O site é seguro para compras?",
        "Vocês fazem entregas aos sábados?",
        "Como entrar em contato com o suporte?",
        "Como recebo a segunda via do boleto?",        # Pergunta repetida
        "Vocês fazem entregas internacionais?",
        "Quais são os benefícios do clube de vantagens?",  # Pergunta repetida
        "Como consultar meu saldo?",
        "Como emitir nota fiscal eletrônica?",
        "Quais cursos online vocês oferecem?",
        "Como faço para agendar uma visita técnica?",
        "Como posso alterar meu plano?",
        "Esqueci minha senha, como recupero?",
        "Como faço uma reclamação na ouvidoria?",
        "Quais são os canais de atendimento ao cliente?",
        "Como faço para cancelar minha assinatura?",   # Pergunta repetida
        "Como atualizar meus dados cadastrais?",       # Pergunta repetida
        "Existe parcelamento sem juros?",              # Pergunta repetida
        "Quais formas de pagamento aceitam?",          # Pergunta repetida
        "Como entrar em contato com o suporte?"        # Pergunta repetida
    ]
    for idx, pergunta_user in enumerate(perguntas, 1):
        print(f"\n[{idx}] Usuário pergunta: '{pergunta_user}'")
        # 1. Verificar se pergunta semelhante já foi respondida:
        match = buscar_similar(pergunta_user)
        if match:
            print("→ Buscando no banco vetorizado (FAQ/HISTÓRICO):")
            print(f"✔ Resposta recuperada do FAQ: '{match[0]['resposta']}' [Confiança: {match[1]:.2f}]")
        else:
            print("→ Nenhuma resposta semelhante encontrada no FAQ. Roteando para modelo LLM...")
            # 2. Roteamento para modelo adequado
            modelo_escolhido = router_modelo(pergunta_user)
            print(f"→ Modelo selecionado: {modelo_escolhido}")
            # 3. Chamar modelo via OpenRouter (simulado)
            resposta = chamar_modelo(modelo_escolhido, pergunta_user)
            # 4. Retornar resposta ao usuário
            print(f"✔ Resposta gerada pelo modelo: {resposta}")
            # 5. Armazenar interação no banco vetorizado
            armazenar_interacao(pergunta_user, resposta, modelo_escolhido)
        time.sleep(5)  # Aguarda 5 segundos antes de processar a próxima pergunta


