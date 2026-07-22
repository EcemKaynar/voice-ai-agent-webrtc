import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from services.topic_router_service import (
    infer_query_topic,
    infer_chunk_topic,
    should_allow_chunk_for_topic
)

load_dotenv()

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "data/qdrant")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "garenta_knowledge_base")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "0"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.28"))

INDEX_META_PATH = Path(QDRANT_LOCAL_PATH) / "_kb_index_meta.json"
CHUNKING_VERSION = 4

_embedder = None


def normalize_space(text):
    text = str(text or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_document_text(text):
    lines = [
        normalize_space(line)
        for line in str(text or "").splitlines()
    ]

    return "\n".join(line for line in lines if line).strip()


def normalize_for_search(text):
    text = str(text or "").lower()

    replacements = {
        "ı": "i",
        "İ": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_embedder():
    global _embedder

    if _embedder is None:
        print(f"Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedder


def embed_texts(texts):
    model = get_embedder()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return [vector.tolist() for vector in vectors]


def get_qdrant_client():
    Path(QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)

    return QdrantClient(path=QDRANT_LOCAL_PATH)


def read_docx_text(file_path):
    file_path = Path(file_path)

    with zipfile.ZipFile(file_path) as docx_zip:
        xml_content = docx_zip.read("word/document.xml")

    root = ET.fromstring(xml_content)

    namespace = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    }

    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        texts = []

        for text_node in paragraph.findall(".//w:t", namespace):
            if text_node.text:
                texts.append(text_node.text)

        paragraph_text = normalize_space("".join(texts))

        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n".join(paragraphs)


def read_text_file(file_path):
    file_path = Path(file_path)

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="cp1254", errors="ignore")


def load_knowledge_documents():
    kb_dir = Path(KNOWLEDGE_BASE_DIR)
    kb_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = [".docx", ".txt", ".md"]
    documents = []

    for file_path in kb_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            if file_path.suffix.lower() == ".docx":
                text = read_docx_text(file_path)
            else:
                text = read_text_file(file_path)

            text = normalize_document_text(text)

            if not text:
                continue

            documents.append({
                "source": file_path.name,
                "path": str(file_path),
                "text": text,
                "size": file_path.stat().st_size,
                "modified_at": int(file_path.stat().st_mtime)
            })

        except Exception as error:
            print(f"Knowledge document okunamadı: {file_path} | Error: {error}")

    return documents


def guess_title_from_text(text, fallback_title):
    text = str(text or "").strip()

    lines = [
        normalize_space(line)
        for line in text.splitlines()
        if normalize_space(line)
    ]

    for line in lines[:5]:
        label_match = re.match(r"^([^:]{4,90}):\s+", line)

        if label_match:
            return label_match.group(1).strip()

        if 4 <= len(line) <= 90:
            return line

    first_sentence = re.split(r"[.!?]", normalize_space(text))[0].strip()

    if 4 <= len(first_sentence) <= 90:
        return first_sentence

    return fallback_title


def looks_like_section_heading(text):
    text = normalize_space(text)

    return bool(
        4 <= len(text) <= 90
        and len(text.split()) <= 10
        and text[-1] not in ".,;!?"
    )


def split_text_into_chunks(text, chunk_size=900, overlap=0):
    text = str(text or "").strip()

    raw_paragraphs = [
        normalize_space(item)
        for item in re.split(r"\n+", text)
        if normalize_space(item)
    ]

    chunks = []
    current = ""

    for paragraph in raw_paragraphs:
        is_section_heading = looks_like_section_heading(paragraph)

        if is_section_heading and current:
            current_lines = [
                line
                for line in current.splitlines()
                if normalize_space(line)
            ]

            current_is_heading_chain = bool(
                current_lines
                and all(looks_like_section_heading(line) for line in current_lines)
            )

            if current_is_heading_chain:
                current = f"{current}\n{paragraph}"
                continue

            chunks.append(current)
            current = paragraph
            continue

        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n{paragraph}"

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    final_chunks = []

    for chunk in chunks:
        chunk = normalize_space(chunk)

        if len(chunk) <= chunk_size * 1.3:
            final_chunks.append(chunk)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        temp = ""

        for sentence in sentences:
            sentence = normalize_space(sentence)

            if not sentence:
                continue

            candidate = f"{temp} {sentence}".strip()

            if len(candidate) <= chunk_size:
                temp = candidate
            else:
                if temp:
                    final_chunks.append(temp)

                temp = sentence

        if temp:
            final_chunks.append(temp)

    return final_chunks


def extract_segment_table_data(text):
    text = normalize_space(text)

    segment_names = ["Ekonomi", "Konfor", "Prestij", "Lüks"]

    segment_data = {
        segment: {}
        for segment in segment_names
    }

    age_pattern = re.compile(
        r"(Ekonomi|Konfor|Prestij|Lüks)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        re.IGNORECASE
    )

    for match in age_pattern.finditer(text):
        segment = match.group(1)

        segment_data[segment]["minimum_surucu_yasi"] = match.group(2)
        segment_data[segment]["genc_surucu_yasi"] = match.group(3)
        segment_data[segment]["minimum_ehliyet_yili"] = match.group(4)
        segment_data[segment]["genc_surucu_ehliyet_yili"] = match.group(5)

    findeks_section_match = re.search(
        r"Segment\s+Findeks\*?(.*?)(Kiralama Süresi|Olası gecikmeler|Yakıt|Araç Teslimi)",
        text,
        re.IGNORECASE
    )

    if findeks_section_match:
        findeks_section = findeks_section_match.group(1)

        findeks_pattern = re.compile(
            r"(Ekonomi|Konfor|Prestij|Lüks)\s+(\d+[.,]?\d*)",
            re.IGNORECASE
        )

        for match in findeks_pattern.finditer(findeks_section):
            segment = match.group(1)
            segment_data[segment]["findeks"] = match.group(2)

    km_section_match = re.search(
        r"Araç Grupları Kilometre Sınırları(.*?)(Ödeme Koşulları|Güvenlik|Rezervasyon)",
        text,
        re.IGNORECASE
    )

    if km_section_match:
        km_section = km_section_match.group(1)

        km_pattern = re.compile(
            r"(Ekonomi|Konfor|Prestij|Lüks)\s+(\d+)\s*TL\s+(\d+[.,]?\d*)\s*TL\s+(\d+)\s*km\s+(\d+)\s*km",
            re.IGNORECASE
        )

        for match in km_pattern.finditer(km_section):
            segment = match.group(1)

            segment_data[segment]["teminat_tutari"] = match.group(2)
            segment_data[segment]["km_asim_bedeli"] = match.group(3)
            segment_data[segment]["gunluk_km_limiti"] = match.group(4)
            segment_data[segment]["aylik_km_limiti"] = match.group(5)

    return segment_data


def build_structured_segment_chunks(document):
    segment_data = extract_segment_table_data(document["text"])
    chunks = []

    for segment, data in segment_data.items():
        if not data:
            continue

        sentences = []

        if (
            data.get("minimum_surucu_yasi")
            and data.get("genc_surucu_yasi")
            and data.get("minimum_ehliyet_yili")
            and data.get("genc_surucu_ehliyet_yili")
        ):
            sentences.append(
                f"{segment} segment yaş ve ehliyet koşulları: "
                f"minimum sürücü yaşı {data.get('minimum_surucu_yasi')}, "
                f"genç sürücü yaşı {data.get('genc_surucu_yasi')}, "
                f"minimum ehliyet yılı {data.get('minimum_ehliyet_yili')}, "
                f"genç sürücü ehliyet yılı {data.get('genc_surucu_ehliyet_yili')}."
            )

        if data.get("findeks"):
            sentences.append(
                f"{segment} segment Findeks koşulu {data.get('findeks')} olarak belirtilmiştir."
            )

        if (
            data.get("teminat_tutari")
            and data.get("km_asim_bedeli")
            and data.get("gunluk_km_limiti")
            and data.get("aylik_km_limiti")
        ):
            sentences.append(
                f"{segment} segment kilometre ve teminat koşulları: "
                f"teminat tutarı {data.get('teminat_tutari')} TL, "
                f"kilometre aşım bedeli {data.get('km_asim_bedeli')} TL, "
                f"günlük kilometre limiti {data.get('gunluk_km_limiti')} km, "
                f"aylık kilometre limiti {data.get('aylik_km_limiti')} km."
            )

        if not sentences:
            continue

        content = " ".join(sentences)

        chunks.append({
    "source": document["source"],
    "path": document["path"],
    "title": f"Segment koşulları - {segment}",
    "content": content,
    "char_count": len(content),
    "structured": True,
    "segment": segment,
    "topic": "segment_conditions",
    "topic_score": 100
})

    return chunks


def build_chunks_from_documents(documents):
    all_chunks = []
    chunk_counter = 0

    for document in documents:
        structured_chunks = build_structured_segment_chunks(document)

        for structured_chunk in structured_chunks:
            chunk_counter += 1
            structured_chunk["chunk_id"] = chunk_counter
            all_chunks.append(structured_chunk)

        chunks = split_text_into_chunks(
            document["text"],
            chunk_size=RAG_CHUNK_SIZE,
            overlap=RAG_CHUNK_OVERLAP
        )

        for index, chunk_text in enumerate(chunks):
            chunk_counter += 1

            title = guess_title_from_text(
                text=chunk_text,
                fallback_title=f"{document['source']} - Parça {index + 1}"
            )

            topic_info = infer_chunk_topic(
            title=title,
            content=chunk_text,
            structured=False,
            segment=None
)

            all_chunks.append({
        "chunk_id": chunk_counter,
         "source": document["source"],
            "path": document["path"],
    "title": title,
    "content": chunk_text,
    "char_count": len(chunk_text),
    "structured": False,
    "segment": None,
    "topic": topic_info.get("topic"),
    "topic_score": topic_info.get("score", 0)
})

    return all_chunks


def build_documents_hash(documents):
    hash_input = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "chunking_version": CHUNKING_VERSION,
        "documents": [
            {
                "path": item["path"],
                "size": item["size"],
                "modified_at": item["modified_at"]
            }
            for item in documents
        ]
    }

    raw = json.dumps(hash_input, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def collection_exists(client, collection_name):
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception:
        return False


def get_collection_count(client, collection_name):
    try:
        count_result = client.count(
            collection_name=collection_name,
            exact=True
        )

        return int(count_result.count)
    except Exception:
        return 0


def load_index_meta():
    if not INDEX_META_PATH.exists():
        return None

    try:
        return json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_index_meta(meta):
    INDEX_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def recreate_collection(client, vector_size):
    if collection_exists(client, QDRANT_COLLECTION):
        client.delete_collection(collection_name=QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )


def index_knowledge_base(force=False):
    started_at = time.perf_counter()

    documents = load_knowledge_documents()
    documents_hash = build_documents_hash(documents)

    client = get_qdrant_client()
    existing_meta = load_index_meta()
    exists = collection_exists(client, QDRANT_COLLECTION)
    existing_count = get_collection_count(client, QDRANT_COLLECTION) if exists else 0

    if (
        not force
        and exists
        and existing_count > 0
        and existing_meta
        and existing_meta.get("documents_hash") == documents_hash
    ):
        return {
            "success": True,
            "indexed": False,
            "message": "Knowledge base index zaten güncel.",
            "collection": QDRANT_COLLECTION,
            "document_count": len(documents),
            "chunk_count": existing_count,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "qdrant_path": QDRANT_LOCAL_PATH,
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000)
        }

    chunks = build_chunks_from_documents(documents)

    if not chunks:
        recreate_collection(
            client=client,
            vector_size=384
        )

        save_index_meta({
            "documents_hash": documents_hash,
            "document_count": len(documents),
            "chunk_count": 0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunking_version": CHUNKING_VERSION,
            "updated_at": time.time()
        })

        return {
            "success": True,
            "indexed": True,
            "message": "Knowledge base boş. Index boş oluşturuldu.",
            "collection": QDRANT_COLLECTION,
            "document_count": len(documents),
            "chunk_count": 0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "qdrant_path": QDRANT_LOCAL_PATH,
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000)
        }

    sample_vector = embed_texts(["test"])[0]
    vector_size = len(sample_vector)

    recreate_collection(
        client=client,
        vector_size=vector_size
    )

    texts = [
        f"{chunk['title']}\n{chunk['content']}"
        for chunk in chunks
    ]

    vectors = embed_texts(texts)

    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "path": chunk["path"],
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "char_count": chunk["char_count"],
                    "structured": chunk.get("structured", False),
                    "segment": chunk.get("segment"),
                    "topic": chunk.get("topic"),
                    "topic_score": chunk.get("topic_score", 0)
                }
            )
        )

    batch_size = 64

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch
        )

    save_index_meta({
        "documents_hash": documents_hash,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_size": vector_size,
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "chunking_version": CHUNKING_VERSION,
        "updated_at": time.time()
    })

    return {
        "success": True,
        "indexed": True,
        "message": "Knowledge base semantic RAG index oluşturuldu.",
        "collection": QDRANT_COLLECTION,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_size": vector_size,
        "qdrant_path": QDRANT_LOCAL_PATH,
        "elapsed_ms": int((time.perf_counter() - started_at) * 1000)
    }


def build_context_from_results(results):
    if not results:
        return ""

    parts = []

    for index, item in enumerate(results, start=1):
        parts.append(
            f"KAYNAK {index}\n"
            f"Başlık: {item.get('title')}\n"
            f"Dosya: {item.get('source')}\n"
            f"Benzerlik Skoru: {item.get('score')}\n"
            f"İçerik:\n{item.get('content')}"
        )

    return "\n\n---\n\n".join(parts)


def to_positive_int(value, default=4):
    if isinstance(value, (tuple, list)):
        if value:
            value = value[0]
        else:
            value = default

    try:
        value = int(value)
    except Exception:
        value = default

    if value < 1:
        value = default

    return value


def query_qdrant(client, query_vector, top_k):
    limit_value = to_positive_int(top_k, default=4)

    try:
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            with_payload=True,
            limit=limit_value
        )

        return response.points

    except AttributeError:
        return client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            with_payload=True,
            limit=limit_value
        )

def search_knowledge(query, top_k=4):
    started_at = time.perf_counter()
    query = normalize_space(query)
    top_k = to_positive_int(top_k, default=4)

    if not query:
        return {
            "found": False,
            "query": query,
            "provider": "qdrant_semantic_rag",
            "results": [],
            "context": "",
            "intent_profiles": [],
            "error": "Query boş."
        }

    index_status = index_knowledge_base(force=False)

    client = get_qdrant_client()

    if not collection_exists(client, QDRANT_COLLECTION):
        return {
            "found": False,
            "query": query,
            "provider": "qdrant_semantic_rag",
            "results": [],
            "context": "",
            "intent_profiles": [],
            "error": "Qdrant collection bulunamadı.",
            "index_status": index_status
        }

    query_norm = normalize_for_search(query)
    topic_result = infer_query_topic(query)
    query_topic = topic_result.get("topic")

    intent_profiles = []

    if query_topic:
        intent_profiles.append(query_topic)

    expanded_query = topic_result.get("expanded_query") or ""
    retrieval_query = f"{query} {expanded_query}".strip()

    query_vector = embed_texts([retrieval_query])[0]
    search_limit = max(top_k * 10, 40)

    raw_points = query_qdrant(
        client=client,
        query_vector=query_vector,
        top_k=search_limit
    )

    query_terms = [
        term
        for term in query_norm.split()
        if len(term) >= 4
    ]

    results = []

    for point in raw_points:
        payload = point.payload or {}
        semantic_score = float(getattr(point, "score", 0) or 0)

        title = payload.get("title") or ""
        content = payload.get("content") or ""
        segment = payload.get("segment")
        structured = bool(payload.get("structured"))
        chunk_topic = payload.get("topic")
        chunk_topic_score = payload.get("topic_score", 0)

        searchable_text = normalize_for_search(f"{title} {content}")
        lexical_score = 0

        for term in query_terms:
            if term in searchable_text:
                lexical_score += 0.05

        if query_topic and chunk_topic == query_topic:
            lexical_score += 0.65

        if query_topic and chunk_topic and query_topic != chunk_topic:
            lexical_score -= 0.25

        if "segment" in query_norm and "segment" in searchable_text:
            lexical_score += 0.12

        if "luks" in query_norm and normalize_for_search(segment or "") == "luks":
            lexical_score += 0.45

        if "ekonomi" in query_norm and normalize_for_search(segment or "") == "ekonomi":
            lexical_score += 0.45

        if "konfor" in query_norm and normalize_for_search(segment or "") == "konfor":
            lexical_score += 0.45

        if "prestij" in query_norm and normalize_for_search(segment or "") == "prestij":
            lexical_score += 0.45

        if structured and query_topic == "segment_conditions":
            lexical_score += 0.35

        combined_score = semantic_score + lexical_score

        if not should_allow_chunk_for_topic(
            query_topic=query_topic,
            chunk_topic=chunk_topic,
            semantic_score=semantic_score,
            lexical_score=lexical_score
        ):
            continue

        if semantic_score < RAG_SCORE_THRESHOLD and lexical_score < 0.25:
            continue

        results.append({
            "chunk_id": payload.get("chunk_id"),
            "title": title,
            "source": payload.get("source"),
            "path": payload.get("path"),
            "content": content,
            "score": round(combined_score, 4),
            "semantic_score": round(semantic_score, 4),
            "lexical_score": round(lexical_score, 4),
            "provider": "qdrant_semantic_rag",
            "structured": structured,
            "segment": segment,
            "topic": chunk_topic,
            "topic_score": chunk_topic_score
        })

    results = sorted(
        results,
        key=lambda item: item.get("score", 0),
        reverse=True
    )[:top_k]

    context = build_context_from_results(results)

    return {
        "found": len(results) > 0,
        "query": query,
        "provider": "qdrant_semantic_rag",
        "collection": QDRANT_COLLECTION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "score_threshold": RAG_SCORE_THRESHOLD,
        "topic": query_topic,
        "topic_result": topic_result,
        "results": results,
        "context": context,
        "intent_profiles": intent_profiles,
        "index_status": index_status,
        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "error": None
    }


def sentence_split(text):
    text = normalize_space(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        normalize_space(sentence)
        for sentence in sentences
        if len(normalize_space(sentence)) >= 20
    ]


def is_table_like_text(text):
    text = normalize_space(text)
    normalized = normalize_for_search(text)

    table_keywords = [
        "segment minimum",
        "ekonomi konfor prestij luks",
        "findeks ekonomi",
        "minimum surucu yasi",
        "genc surucu minimum",
        "gunluk paket",
        "aylik paket"
    ]

    if any(keyword in normalized for keyword in table_keywords):
        return True

    digit_count = sum(1 for char in text if char.isdigit())
    word_count = len(text.split())

    if word_count > 0 and digit_count / word_count > 0.6:
        return True

    segment_word_count = sum(
        1
        for word in ["Ekonomi", "Konfor", "Prestij", "Lüks", "Luxury", "Premium"]
        if word in text
    )

    if segment_word_count >= 3 and digit_count >= 6:
        return True

    return False


def clean_extracted_answer(text):
    text = normalize_space(text)

    remove_fragments = [
        "KİRACI ve ek sürücüler,",
        "KİRACI,",
        "KİRALAYAN'a,"
    ]

    for fragment in remove_fragments:
        text = text.replace(fragment, "")

    text = normalize_space(text)

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if text and text[-1] not in [".", "?", "!"]:
        text += "."

    return text

def build_extractive_rag_fallback(user_query, knowledge_result=None):
    started_at = time.perf_counter()

    def make_result(answer):
        answer = clean_extracted_answer(answer)

        return {
            "success": True,
            "answer": answer,
            "llm_model": "extractive_rag_fallback",
            "llm_first_token_ms": 0,
            "llm_total_ms": int((time.perf_counter() - started_at) * 1000),
            "error": None
        }

    if not knowledge_result or not knowledge_result.get("results"):
        return None

    query_norm = normalize_for_search(user_query)

    query_topic = knowledge_result.get("topic")

    if not query_topic:
        intent_profiles = knowledge_result.get("intent_profiles") or []
        if intent_profiles:
            query_topic = intent_profiles[0]

    results = knowledge_result.get("results", [])

    query_terms = [
        term
        for term in query_norm.split()
        if len(term) >= 4
    ]

    asks_phone = any(
        word in query_norm
        for word in [
            "numara",
            "telefon",
            "hat",
            "hatti",
            "yardim",
            "iletisim",
            "arama",
            "ararim"
        ]
    )

    asks_card_payment = query_topic == "payment" or any(
        phrase in query_norm
        for phrase in [
            "banka kart",
            "sanal kart",
            "debit kart",
            "kart kullan",
            "kredi kart",
            "odeme",
            "odemeyi"
        ]
    )

    asks_late_return = query_topic == "late_return" or any(
        phrase in query_norm
        for phrase in [
            "gec teslim",
            "gec iade",
            "gec kal",
            "gecikme",
            "gec getir"
        ]
    )

    asks_additional_driver = query_topic == "additional_driver" or any(
        phrase in query_norm
        for phrase in [
            "ek surucu",
            "surucu ekle",
            "baska surucu",
            "ikinci surucu",
            "araci baskasi",
            "araci kim kullanabilir",
            "arabayi baskasi"
        ]
    )

    asks_young_driver = (
        query_topic == "segment_conditions"
        and "genc surucu" in query_norm
    )

    specific_segment_terms = ["luks", "ekonomi", "konfor", "prestij"]
    asks_specific_segment = any(term in query_norm for term in specific_segment_terms)

    # Telefon / yardım hattı sorularında direkt numarayı bul.
    if asks_phone:
        for result in results:
            combined_text = f"{result.get('title') or ''}. {result.get('content') or ''}"

            phone_match = re.search(
                r"\b0\s?\d{3}\s?\d{3}\s?\d\s?\d{3}\b|\b0\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b",
                combined_text
            )

            if phone_match:
                number = re.sub(r"\s+", " ", phone_match.group(0)).strip()
                return make_result(
                    f"Dokümanda geçen ilgili numara {number}. Bu numara yol yardım veya acil destek gereken durumlarda kullanılabilir."
                )

    # Genel genç sürücü sorusunda tek segment değil, tüm segmentleri özetle.
    if asks_young_driver and not asks_specific_segment:
        segment_order = ["Ekonomi", "Konfor", "Prestij", "Lüks"]
        segment_conditions = {}

        for result in results:
            title = result.get("title") or ""
            content = result.get("content") or ""

            if not title.startswith("Segment koşulları"):
                continue

            match = re.search(
                r"(Ekonomi|Konfor|Prestij|Lüks) segment yaş ve ehliyet koşulları: "
                r"minimum sürücü yaşı (\d+), genç sürücü yaşı (\d+), "
                r"minimum ehliyet yılı (\d+), genç sürücü ehliyet yılı (\d+)",
                content
            )

            if not match:
                continue

            segment = match.group(1)

            segment_conditions[segment] = {
                "genc_surucu_yasi": match.group(3),
                "genc_surucu_ehliyet_yili": match.group(5)
            }

        if segment_conditions:
            parts = []

            for segment in segment_order:
                data = segment_conditions.get(segment)

                if not data:
                    continue

                parts.append(
                    f"{segment} için {data['genc_surucu_yasi']} yaş ve "
                    f"{data['genc_surucu_ehliyet_yili']} yıl ehliyet"
                )

            if parts:
                return make_result(
                    "Genç sürücü koşulu segmente göre değişir. "
                    + "; ".join(parts)
                    + " şartı aranır."
                )

    # Belirli segment sorusunda structured chunk'ın tamamını kullan.
    top_result = results[0]
    top_title = top_result.get("title") or ""
    top_content = top_result.get("content") or ""
    top_structured = bool(top_result.get("structured"))

    is_segment_question = (
        query_topic == "segment_conditions"
        or any(
            word in query_norm
            for word in [
                "segment",
                "sart",
                "sartlari",
                "kosul",
                "kosullari",
                "ehliyet",
                "yas",
                "findeks",
                "teminat",
                "kilometre",
                "km"
            ]
        )
    )

    if (
        top_content
        and not asks_additional_driver
        and not asks_late_return
        and is_segment_question
        and (
            top_structured
            or top_title.startswith("Segment koşulları")
        )
    ):
        return make_result(normalize_space(top_content))

    # Ek sürücü sorusunda segment/yaş/ehliyet chunk'larına kayma.
    if asks_additional_driver:
        selected_units = []

        for result in results:
            title = result.get("title") or ""
            content = result.get("content") or ""
            title_norm = normalize_for_search(title)

            if title_norm.startswith("segment kosullari"):
                continue

            units = re.split(
                r"(?<=[.!?])\s+|;\s+|\n+|•\s*|-{2,}",
                content
            )

            for unit in units:
                unit = normalize_space(unit)
                unit_norm = normalize_for_search(unit)

                if len(unit) < 25:
                    continue

                if "ek surucu" not in unit_norm and "sozlesme ve teslimat" not in unit_norm:
                    continue

                if any(
                    bad in unit_norm
                    for bad in [
                        "findeks",
                        "minimum surucu yasi",
                        "ehliyet yili",
                        "araca zarar verecek",
                        "yuklemenin haddini",
                        "tasinmasinda",
                        "karayollari trafik kanunu"
                    ]
                ):
                    continue

                selected_units.append(unit)

        selected_units = list(dict.fromkeys(selected_units))

        if selected_units:
            answer = " ".join(selected_units[:2])

            if "5 adet ek sürücü" not in answer and "5 adet ek surucu" not in normalize_for_search(answer):
                for result in results:
                    content = result.get("content") or ""
                    if "5 adet ek sürücü" in content or "5 adet ek surucu" in normalize_for_search(content):
                        units = re.split(r"(?<=[.!?])\s+", content)
                        for unit in units:
                            if "5 adet ek" in normalize_for_search(unit):
                                answer = f"{answer} {normalize_space(unit)}"
                                break
                        break

            return make_result(answer)

    # Geç teslim sorusunda doğru gecikme paragrafını direkt kullan.
    if asks_late_return:
        for result in results:
            title = result.get("title") or ""
            content = result.get("content") or ""
            combined_norm = normalize_for_search(f"{title} {content}")

            if (
                "olasi gecikmeler" in combined_norm
                and "2 saat" in combined_norm
                and "3 saat" in combined_norm
                and "4 saat" in combined_norm
            ):
                return make_result(content)

        selected_units = []

        for result in results:
            content = result.get("content") or ""

            units = re.split(
                r"(?<=[.!?])\s+|;\s+|\n+|•\s*|-{2,}",
                content
            )

            for unit in units:
                unit = normalize_space(unit)
                unit_norm = normalize_for_search(unit)

                if len(unit) < 25:
                    continue

                if any(
                    bad in unit_norm
                    for bad in [
                        "yakit",
                        "depo",
                        "eksik yakit",
                        "no show",
                        "teslim alinmamasi"
                    ]
                ):
                    continue

                if any(
                    good in unit_norm
                    for good in [
                        "gecikme",
                        "2 saat",
                        "3 saat",
                        "4 saat",
                        "kira bedeli",
                        "gunluk kira",
                        "1 3",
                        "2 3"
                    ]
                ):
                    selected_units.append(unit)

        selected_units = list(dict.fromkeys(selected_units))

        if selected_units:
            return make_result(" ".join(selected_units[:3]))

    # Ödeme / banka kartı sorusunda alakasız üyelik saklama metnini alma.
    if asks_card_payment:
        selected_units = []

        for result in results:
            title = result.get("title") or ""
            content = result.get("content") or ""

            title_clean = normalize_space(title)
            title_norm = normalize_for_search(title_clean)

            if (
                title_clean
                and "banka kart" in title_norm
                and "sanal kart" in title_norm
                and "gecerli degildir" in title_norm
            ):
                selected_units.append(title_clean)

            units = re.split(
                r"(?<=[.!?])\s+|;\s+|\n+|•\s*|-{2,}",
                content
            )

            for unit in units:
                unit = normalize_space(unit)
                unit_norm = normalize_for_search(unit)

                if len(unit) < 25:
                    continue

                if any(
                    bad in unit_norm
                    for bad in [
                        "silme talebiniz",
                        "uyelik sozlesmesi",
                        "6 ay sureyle",
                        "ssl guvenlik",
                        "3d secure",
                        "rezervasyonunuz",
                        "no show"
                    ]
                ):
                    continue

                if any(
                    good in unit_norm
                    for good in [
                        "banka kart",
                        "sanal kart",
                        "gecerli degildir",
                        "sahsi kredi kart",
                        "kredi kartindan tahsil",
                        "odeme kosullari"
                    ]
                ):
                    selected_units.append(unit)

        selected_units = list(dict.fromkeys(selected_units))

        if selected_units:
            return make_result(" ".join(selected_units[:2]))

    # Genel fallback: kalan durumlarda en iyi cümleleri seç.
    all_units = []

    for result in results:
        content = result.get("content") or ""
        title = result.get("title") or ""
        result_score = float(result.get("score") or 0)

        if not content and not title:
            continue

        candidate_units = []

        title_clean = normalize_space(title)

        if (
            title_clean
            and len(title_clean) >= 20
            and "parça" not in normalize_for_search(title_clean)
            and ".docx" not in title_clean.lower()
        ):
            candidate_units.append({
                "unit": title_clean,
                "source_type": "title"
            })

        raw_units = re.split(
            r"(?<=[.!?])\s+|;\s+|\n+|•\s*|-{2,}",
            content
        )

        for unit in raw_units:
            unit = normalize_space(unit)

            if len(unit) < 25:
                continue

            candidate_units.append({
                "unit": unit,
                "source_type": "content"
            })

        for candidate in candidate_units:
            unit = candidate["unit"]
            source_type = candidate["source_type"]

            if is_table_like_text(unit):
                continue

            all_units.append({
                "unit": unit,
                "title": title,
                "source_type": source_type,
                "result_score": result_score
            })

    if not all_units:
        return None

    scored_units = []

    for item in all_units:
        unit = item["unit"]
        title = item["title"]

        unit_norm = normalize_for_search(unit)
        title_norm = normalize_for_search(title)

        overlap = sum(
            1
            for term in query_terms
            if term in unit_norm or term in title_norm
        )

        exact_bonus = 0

        if item["source_type"] == "title":
            exact_bonus += 2

        number_bonus = 2 if re.search(r"\d", unit) else 0
        short_bonus = 1 if len(unit) <= 280 else 0

        final_score = overlap + exact_bonus + number_bonus + short_bonus + item["result_score"]

        scored_units.append({
            "unit": unit,
            "score": final_score
        })

    selected = []
    seen_units = set()

    for item in sorted(scored_units, key=lambda x: x["score"], reverse=True):
        if item["score"] <= 0:
            continue

        normalized_unit = normalize_for_search(item["unit"])

        if normalized_unit in seen_units:
            continue

        seen_units.add(normalized_unit)
        selected.append(item)

        if len(selected) >= 2:
            break

    if not selected:
        selected = sorted(scored_units, key=lambda x: x["score"], reverse=True)[:1]

    answer = " ".join(item["unit"] for item in selected)

    if len(answer) > 520:
        answer = answer[:520].rsplit(" ", 1)[0].strip() + "."

    return make_result(answer)


def get_knowledge_base_status():
    documents = load_knowledge_documents()
    meta = load_index_meta()

    try:
        client = get_qdrant_client()
        exists = collection_exists(client, QDRANT_COLLECTION)
        count = get_collection_count(client, QDRANT_COLLECTION) if exists else 0
    except Exception as error:
        exists = False
        count = 0
        meta = meta or {}
        meta["qdrant_error"] = str(error)

    return {
        "enabled": True,
        "provider": "qdrant_semantic_rag",
        "knowledge_base_dir": KNOWLEDGE_BASE_DIR,
        "qdrant_path": QDRANT_LOCAL_PATH,
        "collection": QDRANT_COLLECTION,
        "collection_exists": exists,
        "indexed_chunk_count": count,
        "document_count": len(documents),
        "documents": [
            {
                "source": item["source"],
                "path": item["path"],
                "size": item["size"],
                "modified_at": item["modified_at"]
            }
            for item in documents
        ],
        "embedding_model": EMBEDDING_MODEL_NAME,
        "score_threshold": RAG_SCORE_THRESHOLD,
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "chunking_version": CHUNKING_VERSION,
        "index_meta": meta
    }


def reload_knowledge_base():
    return index_knowledge_base(force=True)