"""
Document Query Endpoints.

Provides document retrieval and query testing functionality:
- List documents in collection
- Test queries with detailed debugging
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Form
from src.core.exceptions import ChromaDBError, ValidationError, CollectionNotFoundError
from typing import List, Dict, Any, Optional
import logging
import asyncio

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.core.performance_monitor import perf_monitor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/collections/{collection_name}/documents")
async def get_documents(
    collection_name: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Get documents from collection with pagination."""
    logger.debug(f"Getting documents from '{collection_name}': limit={limit}, offset={offset}")

    try:
        docs_data = await rag_client.get_documents(
            collection_name=collection_name,
            limit=limit,
            offset=offset
        )
        logger.debug(f"Retrieved {len(docs_data.get('documents', []))} documents")
        return docs_data

    except Exception as e:
        logger.error(f"Failed to get documents: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.post("/query/test")
async def test_query(
    query: str = Form(...),
    collection_name: str = Form("default"),
    n_results: int = Form(5),
    generate_answer: bool = Form(False),
    advanced_mode: bool = Form(False),  # NEW: Enable Multi-Collection like DraftService
    domain: Optional[str] = Form(None),  # NEW: Domain-based routing
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """
    Test RAG query with custom parameters and optional LLM answer generation.

    NEW ADVANCED MODE:
    - advanced_mode=True: Uses SAME Multi-Collection logic as DraftService
    - Queries ALL collections (or domain-filtered collections)
    - Cross-collection ranking (Top N most relevant across ALL collections)
    - Returns collection stats showing source breakdown
    """
    logger.debug(f"Testing query: {query[:50]}... (advanced={advanced_mode}, domain={domain}, generate_answer={generate_answer})")

    with perf_monitor.track_operation(
        "test_query",
        tags={
            "collection": collection_name if not advanced_mode else "multi-collection",
            "n_results": n_results,
            "advanced_mode": advanced_mode,
            "domain": domain,
            "generate_answer": generate_answer
        }
    ):
        try:
            formatted_results = []
            collection_stats = {}

            if advanced_mode:
                # ✅ ADVANCED MODE: Use SAME logic as DraftService.generate_draft()
                logger.info("🚀 Advanced Mode: Using Multi-Collection RAG (like DraftService)")

                # Step 1: Get all available collections
                response = await rag_client.list_collections()
                if not response.is_success:
                    logger.error(f"Failed to list collections: {response.error}")
                    all_available_collections = []
                else:
                    all_available_collections = response.data

                # Step 2: Domain-based collection selection (SAME as DraftService)
                from src.services.config_service import config_service
                config = config_service.load_configuration()

                # Load rag_domains.json
                import json
                from pathlib import Path
                try:
                    domains_path = Path(__file__).parent.parent.parent / "core" / "rag_domains.json"
                    with open(domains_path, 'r', encoding='utf-8') as f:
                        rag_domains = json.load(f).get("domains", {})
                except Exception:
                    rag_domains = {}

                collections_to_query = []

                if domain and rag_domains:
                    logger.info(f"Domain-based routing for domain: '{domain}'")
                    domain_info = rag_domains.get(domain)
                    if domain_info:
                        domain_collections = domain_info.get("collections", [])
                        collections_to_query = [c for c in domain_collections if c in all_available_collections]
                        if not collections_to_query:
                            logger.warning(f"Domain '{domain}' collections not found. Falling back to all.")

                # Fallback: Use all collections or config-filtered
                if not collections_to_query:
                    config_collections_str = config.get("DRAFT_RAG_COLLECTIONS", "").strip()
                    if config_collections_str:
                        configured_list = [c.strip() for c in config_collections_str.split(",")]
                        collections_to_query = [c for c in configured_list if c in all_available_collections]
                        if not collections_to_query:
                            collections_to_query = all_available_collections
                    else:
                        collections_to_query = all_available_collections

                logger.info(f"Querying {len(collections_to_query)} collections: {collections_to_query}")

                # Step 3: Query each collection with adaptive k (SAME as DraftService)
                all_results = []
                k_per_collection = max(10, 100 // len(collections_to_query)) if collections_to_query else n_results

                for collection_name_iter in collections_to_query:
                    try:
                        response = await rag_client.query(
                            query_text=query,
                            collection_names=[collection_name_iter],
                            n_results=k_per_collection
                        )

                        if response.is_success and response.data:
                            for res in response.data:
                                res['source_collection'] = collection_name_iter
                            all_results.extend(response.data)
                            collection_stats[collection_name_iter] = len(response.data)
                    except Exception as col_error:
                        logger.warning(f"Failed to query collection '{collection_name_iter}': {col_error}")
                        continue

                # Step 4: Cross-collection ranking (SAME as DraftService)
                all_results.sort(key=lambda x: x.get('relevance_score', 1 - x.get('distance', 0)), reverse=True)
                top_results = all_results[:n_results]

                # Step 5: Format results with collection tags
                for i, result in enumerate(top_results):
                    formatted_results.append({
                        "rank": i + 1,
                        "content": result.get('content', result.get('document', '')),
                        "metadata": result.get('metadata', {}),
                        "distance": result.get('distance', 0),
                        "relevance_score": result.get('relevance_score', 1 - result.get('distance', 0)),
                        "source_collection": result.get('source_collection', 'unknown')  # NEW!
                    })

                # Collection breakdown for top results
                top_collection_breakdown = {}
                for res in top_results:
                    coll = res.get('source_collection', 'unknown')
                    top_collection_breakdown[coll] = top_collection_breakdown.get(coll, 0) + 1

                logger.info(
                    f"Advanced Mode: {len(all_results)} total results from {len(collection_stats)} collections. "
                    f"Top {n_results} from: {top_collection_breakdown}"
                )

            else:
                # ❌ SIMPLE MODE: Single-collection query (old behavior)
                logger.info(f"Simple Mode: Single-collection query on '{collection_name}'")

                response = await rag_client.query(
                    query_text=query,
                    collection_names=[collection_name],
                    n_results=n_results
                )

                if not response.is_success:
                    logger.error(f"Query failed: {response.error}")
                    raise ChromaDBError(response.error)

                for i, result in enumerate(response.data):
                    formatted_results.append({
                        "rank": i + 1,
                        "content": result.get('content', result.get('document', '')),
                        "metadata": result.get('metadata', {}),
                        "distance": result.get('distance', 0),
                        "relevance_score": 1 - result.get('distance', 0),
                        "source_collection": collection_name  # Single collection
                    })

                collection_stats[collection_name] = len(formatted_results)

            logger.info(f"Query returned {len(formatted_results)} results")

            result_data = {
                "query": query,
                "collection": collection_name if not advanced_mode else "multi-collection",
                "results": formatted_results,
                "source_chunks": formatted_results,
                "total_results": len(formatted_results),
                "advanced_mode": advanced_mode,
                "collection_stats": collection_stats,  # NEW: Shows which collections were queried
                "domain": domain
            }

            # Generate LLM answer if requested
            if generate_answer and formatted_results:
                try:
                    from src.core.llm_singleton import get_llm
                    llm = get_llm()

                    if llm:
                        # For counting queries, use ALL available results
                        chunks_to_use = formatted_results

                        # Build context from results with collection tags (NEW for Advanced Mode)
                        if advanced_mode:
                            context_chunks = "\n\n---\n\n".join([
                                f"[Collection: {r['source_collection'].upper()} | Chunk {i+1} | Relevance: {r['relevance_score']:.2f}]\n{r['content']}"
                                for i, r in enumerate(chunks_to_use)
                            ])
                            collection_info = f"Daten aus {len(collection_stats)} Collections: {', '.join(collection_stats.keys())}"
                        else:
                            context_chunks = "\n\n---\n\n".join([
                                f"[Chunk {i+1}, Relevance: {r['relevance_score']:.2f}]\n{r['content']}"
                                for i, r in enumerate(chunks_to_use)
                            ])
                            collection_info = f"Daten aus Collection: {collection_name}"

                        # Build prompt
                        num_chunks = len(chunks_to_use)
                        num_requested = n_results

                        prompt = f"""WICHTIG: ANTWORTE AUSSCHLIEẞLICH AUF DEUTSCH! NIEMALS AUF ENGLISCH!

Du bist ein hilfreicher Firmen-Assistent mit Zugriff auf eine Multi-Domain Wissensdatenbank.

═══════════════════════════════════════════════════
RAG-SYSTEM INFO:
{"🚀 ADVANCED MODE: Multi-Collection Query über ALLE Domains" if advanced_mode else "📋 SIMPLE MODE: Single-Collection Query"}
{collection_info}
═══════════════════════════════════════════════════

═══════════════════════════════════════════════════
EINSCHRÄNKUNG DER SUCHERGEBNISSE:
Du siehst die TOP {num_chunks} relevantesten Chunks (Limit: {num_requested}).
{"Diese wurden aus ALLEN verfügbaren Collections ausgewählt und nach Relevanz sortiert." if advanced_mode else ""}
Es können WEITERE Informationen in der Datenbank existieren, die hier NICHT gezeigt werden!
═══════════════════════════════════════════════════

VERFÜGBARE DATEN:
{context_chunks}

KUNDENFRAGE: {query}

═══════════════════════════════════════════════════
DEINE ANWEISUNGEN (STRIKT BEFOLGEN):
═══════════════════════════════════════════════════

1. SPRACHE: Antworte AUSSCHLIEẞLICH auf DEUTSCH!

2. DATENQUELLE NUTZEN:
   {"- Die Chunks kommen aus verschiedenen Collections (siehe Collection-Tag)" if advanced_mode else "- Die Chunks kommen aus einer Collection"}
   - Nutze die Relevanz-Scores - höhere Scores = verlässlichere Daten
   - Bei widersprüchlichen Informationen: Bevorzuge höhere Relevanz

3. ZÄHLEN: Wenn nach "wie viele" gefragt wird:
   - Zähle die UNTERSCHIEDLICHEN Entities in den Chunks
   - Sage: "In den angezeigten Ergebnissen sind mindestens X [Items] sichtbar"
   - Erwähne: "Es könnten weitere [Items] in der Datenbank existieren"

4. EHRLICHKEIT: Wenn das Limit zu niedrig ist, weise darauf hin!

5. DETAILS: Nenne konkrete Details aus den Chunks (Namen, Zahlen, Spezifikationen, etc.)

6. TRANSPARENZ:
   {"- Wenn Daten aus verschiedenen Collections kommen, erkenne das an" if advanced_mode else ""}
   - Sei ehrlich wenn Informationen fehlen oder unklar sind

═══════════════════════════════════════════════════
ANTWORT (AUF DEUTSCH):"""

                        # Generate answer using LlamaIndex LLM interface
                        # LlamaIndex LLMs use different methods than LangChain
                        try:
                            if hasattr(llm, 'complete'):  # Ollama, OpenAI, etc. in LlamaIndex
                                llm_response = await asyncio.to_thread(llm.complete, prompt)
                                answer = str(llm_response)
                            elif hasattr(llm, 'predict'):  # Alternative LlamaIndex method
                                llm_response = await asyncio.to_thread(llm.predict, prompt)
                                answer = str(llm_response)
                            elif hasattr(llm, 'chat'):  # Chat-based LLM
                                llm_response = await asyncio.to_thread(llm.chat, prompt)
                                answer = str(llm_response)
                            else:
                                # Fallback: try calling the LLM directly
                                llm_response = await asyncio.to_thread(llm, prompt)
                                answer = str(llm_response)
                        except Exception as invoke_error:
                            logger.error(f"LLM invocation failed: {invoke_error}")
                            answer = f"LLM invocation error: {str(invoke_error)}"

                        result_data["answer"] = answer
                        logger.info(f"Generated LLM answer ({len(answer)} chars)")
                    else:
                        logger.warning("LLM not available for answer generation")
                        result_data["answer"] = "LLM is not available. Please check your LLM configuration."

                except Exception as llm_error:
                    logger.error(f"LLM answer generation failed: {llm_error}", exc_info=True)
                    result_data["answer"] = f"Failed to generate answer: {str(llm_error)}"

            return result_data

        except Exception as e:
            logger.error(f"Query test failed: {e}", exc_info=True)
            raise ChromaDBError(str(e))
