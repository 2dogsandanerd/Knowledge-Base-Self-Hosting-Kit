import json
from pathlib import Path
from src.core.rag_client import RAGClient
from src.core.llm_singleton import get_llm
from src.services.learning_manager import LearningManager
from src.services.conversation_manager import ConversationManager
from src.core.services.query_service import QueryService # Import QueryService
from loguru import logger
from typing import Optional, Dict, Any, List
from src.core.feature_limits import Edition

class DraftService:
    """
    Draft generation service using RAG + LLM.
    Adapted from Streamlit to FastAPI (SQLAlchemy dependencies).
    """

    def __init__(
        self,
        rag_client: RAGClient,
        query_service: QueryService, # Add QueryService
        learning_manager: LearningManager,
        conversation_manager: Optional[ConversationManager] = None,
        config_override: dict = None,
        edition: Edition = Edition.DEVELOPER
    ):
        self.rag_client = rag_client
        self.query_service = query_service # Store QueryService
        self.learning_manager = learning_manager
        self.conversation_manager = conversation_manager
        self.config = config_override or {}
        self.edition = edition
        self.rag_domains = self._load_rag_domains()

    def _load_rag_domains(self) -> Dict[str, Any]:
        """Loads the RAG domain configuration from JSON file."""
        try:
            # Correctly locate the file relative to this file's location
            domains_path = Path(__file__).parent.parent / "core" / "rag_domains.json"
            with open(domains_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded RAG domains from {domains_path}")
                return data.get("domains", {})
        except FileNotFoundError:
            logger.warning("rag_domains.json not found. Domain-based routing will be disabled.")
            return {}
        except json.JSONDecodeError:
            logger.error("Failed to decode rag_domains.json. Check for syntax errors.")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading rag_domains.json: {e}")
            return {}

    async def generate_draft(
        self,
        email_data: dict,
        user_id: int,
        use_rag: bool = True,
        domain: Optional[str] = None
    ) -> dict:
        """
        Generate draft response using RAG + LLM.

        INTELLIGENT COLLECTION ROUTING:
        - If a 'domain' is provided, queries only collections associated with that domain.
        - Otherwise, queries ALL available ChromaDB collections dynamically (or as filtered by env var).
        - Uses semantic similarity (embeddings) to find relevant documents.
        - Automatically picks the TOP 10 most relevant results across all queried collections.

        Args:
            email_data: Email information (subject, body, sender)
            user_id: User ID for tracking
            use_rag: Whether to use RAG context
            domain: Optional domain to restrict RAG search (for future router use)

        Returns:
            Dict with draft, rag_context, model_used
        """
        try:
            # RAG Status Tracking
            rag_status = "disabled"
            rag_error = None
            rag_collection_count = 0
            rag_result_count = 0

            # Query RAG for context
            rag_context = ""
            if use_rag:
                query = f"{email_data.get('subject', '')} {email_data.get('body', '')}"

                try:
                    # Determine collections to query (Domain-based and Fallback Logic)
                    response = await self.rag_client.list_collections()
                    if not response.is_success:
                        logger.error(f"Failed to list collections: {response.error}")
                        all_available_collections = []
                    else:
                        all_available_collections = response.data

                    collections_to_query: List[str] = []
                    if domain and self.rag_domains:
                        logger.info(f"Attempting to use domain-based routing for domain: '{domain}'")
                        domain_info = self.rag_domains.get(domain)
                        if domain_info:
                            domain_collections = domain_info.get("collections", [])
                            collections_to_query = [c for c in domain_collections if c in all_available_collections]
                            if not collections_to_query:
                                logger.warning(f"Domain '{domain}' specified, but none of its collections {domain_collections} exist. Falling back to default.")
                        else:
                            logger.warning(f"Domain '{domain}' not found in rag_domains.json. Falling back to default behavior.")
                    
                    if not collections_to_query:
                        config_collections_str = self.config.get("DRAFT_RAG_COLLECTIONS", "").strip()
                        if config_collections_str:
                            configured_list = [c.strip() for c in config_collections_str.split(",")]
                            collections_to_query = [c for c in configured_list if c in all_available_collections]
                            if not collections_to_query:
                                logger.warning(f"Configured collections {configured_list} not found. Using all available collections.")
                                collections_to_query = all_available_collections
                        else:
                            collections_to_query = all_available_collections
                    
                    logger.info(f"Final list of collections to be queried: {collections_to_query}")

                    if not collections_to_query:
                        logger.warning("No RAG collections found to query in ChromaDB")
                        rag_status = "no_collections"
                    else:
                        # Call the new centralized answer_query from QueryService
                        rag_response = await self.query_service.answer_query(
                            query_text=query,
                            collection_names=collections_to_query,
                            final_k=10, # Use top 10 results for context
                            system_prompt=None, # System prompt is handled by _build_prompt
                            temperature=0.1 # Default temperature, can be made configurable
                        )

                        # Extract data from the standardized response
                        rag_status = "success" if rag_response["metadata"]["success"] else "failed"
                        rag_error = rag_response["metadata"]["error"]
                        rag_collection_count = len(rag_response["metadata"]["collections_queried"])
                        rag_result_count = rag_response["metadata"]["final_k"]

                        if rag_response["context"]:
                            context_parts = []
                            for node in rag_response["context"]:
                                collection_tag = node.get('source_collection', 'UNKNOWN').upper()
                                relevance = node.get('relevance_score', 0)
                                content = node.get('content', '')
                                context_parts.append(
                                    f"[{collection_tag} - Relevanz: {relevance:.2f}]\n{content}"
                                )
                            rag_context = "\n\n---\n\n".join(context_parts)
                        else:
                            rag_context = ""
                            rag_status = "no_context"
                        
                        logger.info(
                            f"RAG query: {rag_result_count} final results from {rag_collection_count} collections. "
                            f"Status: {rag_status}"
                        )

                except Exception as e:
                    logger.error(f"RAG query failed in DraftService: {e}", exc_info=True)
                    rag_context = ""
                    rag_status = "failed"
                    rag_error = str(e)
            else:
                rag_status = "disabled"
                logger.info("RAG context disabled by request")

            llm = get_llm()
            prompt = self._build_prompt(email_data, rag_context)
            # Use LlamaIndex LLM interface instead of LangChain invoke
            try:
                if hasattr(llm, 'complete'):
                    response = llm.complete(prompt)
                elif hasattr(llm, 'predict'):
                    response = llm.predict(prompt)
                elif hasattr(llm, 'chat'):
                    response = llm.chat(prompt)
                else:
                    response = llm(prompt)  # Direct call as fallback
                draft = str(response)
            except Exception as invoke_error:
                logger.error(f"LLM invocation failed in draft service: {invoke_error}")
                draft = f"LLM invocation error: {str(invoke_error)}"

            if "NO_ANSWER_NEEDED" in draft:
                parts = draft.split("|")
                category = parts[1].strip() if len(parts) > 1 else "UNKNOWN"
                logger.info(f"Email classified as NO_ANSWER_NEEDED: {category}")
                return {
                    "draft": "", "no_answer_needed": True, "reason_category": category,
                    "reason_text": draft, "rag_context": rag_context,
                    "model": getattr(llm, 'model_name', getattr(llm, 'model', 'unknown')),
                    "rag_status": rag_status, "rag_error": rag_error,
                    "rag_collection_count": rag_collection_count, "rag_result_count": rag_result_count
                }

            if draft.startswith("DRAFT |"):
                draft = draft.replace("DRAFT |", "", 1).strip()

            result = {
                "draft": draft, "no_answer_needed": False, "reason_category": None,
                "rag_context": rag_context,
                "model": getattr(llm, 'model_name', getattr(llm, 'model', 'unknown')),
                "rag_status": rag_status, "rag_error": rag_error,
                "rag_collection_count": rag_collection_count, "rag_result_count": rag_result_count
            }
            logger.success(f"Draft generated (length: {len(draft)} chars)")
            return result

        except Exception as e:
            logger.error(f"Draft generation failed: {e}")
            return {"error": str(e)}

    async def generate_draft_with_learning(
        self,
        email_data: dict,
        user_id: int,
        thread_id: str,
        use_rag: bool = True,
        domain: Optional[str] = None
    ) -> dict:
        """
        Generate draft and store in learning DB.
        """
        # Check if learning feature is available based on edition
        if not self._is_learning_enabled():
            logger.info("Learning feature disabled for this edition, generating draft without learning")
            return await self.generate_draft(email_data, user_id, use_rag=use_rag, domain=domain)

        existing_pair = await self.learning_manager.get_pair_by_thread_id(thread_id, user_id)
        if existing_pair:
            logger.warning(f"Thread {thread_id} already has a draft (ID: {existing_pair.id}), returning existing draft")
            return {
                "draft": existing_pair.draft_content, "draft_id": existing_pair.id,
                "already_existed": True, "status": existing_pair.status
            }

        result = await self.generate_draft(email_data, user_id, use_rag=use_rag, domain=domain)

        if "error" in result or result.get("no_answer_needed"):
            return result

        draft_id = self.learning_manager.add_draft(
            user_id=user_id, thread_id=thread_id,
            draft_message_id=f"draft_{thread_id}_{user_id}", draft_content=result["draft"]
        )
        result["draft_id"] = draft_id

        if self.conversation_manager:
            conv_id = self.conversation_manager.store_conversation(
                user_id=user_id, email_data=email_data, generated_response=result["draft"],
                rag_context=result.get("rag_context", ""), model_used=result.get("model", "")
            )
            result["conversation_id"] = conv_id

        return result

    def _is_learning_enabled(self) -> bool:
        """
        Check if learning feature is enabled for the current edition.
        """
        from src.core.feature_limits import FeatureLimits
        return FeatureLimits.is_feature_enabled("learning_system", self.edition)

    def _build_prompt(self, email_data: dict, rag_context: str) -> str:
        """Build prompt for LLM with strong RAG context usage instructions"""
        sender = email_data.get('sender', 'Unknown')
        subject = email_data.get('subject', 'No Subject')
        body = email_data.get('body', '')
        email_lang = self._detect_language(subject + " " + body)

        prompt = f"""Du bist ein professioneller E-Mail-Assistent mit Zugriff auf eine Wissensdatenbank.

=== EINGEHENDE E-MAIL ===
Von: {sender}
Betreff: {subject}
Sprache: {email_lang}

{body}

"""
        if rag_context:
            prompt += f"""
=== WISSENSDATENBANK KONTEXT ===
Die folgenden Informationen wurden aus deiner Wissensdatenbank abgerufen und sind HOCHRELEVANT für diese E-Mail.
Jeder Eintrag ist mit seiner Quell-Collection (z.B. PRODUKTE, KUNDEN) und Relevanz-Score markiert.

{rag_context}

"""

        prompt += f"""
=== DEINE AUFGABE ===
Analysiere die E-Mail und antworte mit EINEM dieser Formate:

**A) NO_ANSWER_NEEDED | NEWSLETTER**
   → Marketing-E-Mails, Werbeinhalte, Newsletter

**B) NO_ANSWER_NEEDED | NOTIFICATION**
   → Automatische Benachrichtigungen, Bestätigungen, Quittungen, Versand-Updates

**C) NO_ANSWER_NEEDED | INSUFFICIENT_CONTEXT**
   → Du hast KEINE relevanten Informationen in der obigen Wissensdatenbank
   → Erfinde NIEMALS Informationen!

**D) NO_ANSWER_NEEDED | OUT_OF_SCOPE**
   → Die Frage liegt außerhalb deiner Expertise oder Wissensdatenbank

**E) DRAFT | [Deine Antwort hier]**
   → E-Mail benötigt eine Antwort UND du hast relevanten Kontext oben

   **KRITISCHE ANWEISUNGEN FÜR ENTWÜRFE:**
   1. **EXAKTE IDENTIFIKATOREN ZUERST ABGLEICHEN!**
      - Wenn E-Mail nach "PROD-O-2024" fragt → suche EXAKTEN Code "PROD-O-2024" im Kontext
      - Wenn E-Mail nach "Kunde #12345" fragt → suche EXAKTE Kunden-ID
      - Exakte ID-Übereinstimmungen haben PRIORITÄT über semantische Ähnlichkeit
      - Wenn exakte Übereinstimmung gefunden → verwende ausschließlich diese Daten

   2. **VERWENDE DIE WISSENSDATENBANK-DATEN AKTIV!**
      - Wenn PRODUKTE-Daten gezeigt werden → zitiere spezifische Produktdetails, Preise, Spezifikationen
      - Wenn KUNDEN-Daten gezeigt werden → verweise auf Kundenhistorie, Status, Präferenzen
      - Zitiere exakte Daten aus dem Kontext (Preise, Teilenummern, Daten, usw.)

   3. **ANTWORTE IN {email_lang.upper()}!**
      - Deutsche E-Mail → Deutsche Antwort
      - Englische E-Mail → Englische Antwort

   4. **SEI SPEZIFISCH, NICHT ALLGEMEIN!**
      ❌ SCHLECHT: "Wir bieten verschiedene Produkte an, die Sie interessieren könnten."
      ✅ GUT: "Basierend auf Ihrer Anfrage wäre unser Produkt XY (Art.Nr. 12345) zu €49,90 ideal, weil..."

   5. **STRUKTUR:**
      - Begrüßung
      - Direkte Antwort unter Verwendung der Wissensdatenbank-Daten
      - Spezifische Produkt-/Kundendetails aus dem Kontext
      - Handlungsaufforderung oder nächste Schritte
      - Professioneller Abschluss

   6. **WENN DIE WISSENSDATENBANK DIE ANTWORT HAT → VERWENDE SIE!**
      Sage nicht "unzureichender Kontext", wenn Produkt-/Kundendaten oben klar gezeigt werden.

**AUSGABEFORMAT:**
[NO_ANSWER_NEEDED | KATEGORIE] oder [DRAFT | Antworttext in {email_lang}]

Antwort:"""
        return prompt

    def _detect_language(self, text: str) -> str:
        """Simple language detection (German vs English)"""
        text_lower = text.lower()
        german_words = ['der', 'die', 'das', 'und', 'oder', 'ich', 'sie', 'haben', 'ist', 'sind',
                       'für', 'mit', 'von', 'zu', 'auf', 'bei', 'über', 'unter', 'nach']
        english_words = ['the', 'and', 'or', 'is', 'are', 'have', 'has', 'for', 'with', 'from',
                        'about', 'this', 'that', 'your', 'our', 'can', 'will', 'would']
        german_count = sum(1 for word in german_words if f' {word} ' in f' {text_lower} ')
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
        if german_count > english_count:
            return "German"
        elif english_count > 0:
            return "English"
        else:
            return "Unknown (defaulting to English)"

    async def filter_emails_batch(self, email_list: list, user_id: int, domain: Optional[str] = None) -> dict:
        """
        Filter multiple emails using LLM classification.
        """
        responses = []
        filtered_emails = []
        draft_needed_emails = []
        
        for email in email_list:
            # Pass domain down to generate_draft for consistent routing
            result = await self.generate_draft(
                {"sender": email.get("sender", ""), "subject": email.get("subject", ""), "body": email.get("body", "")}, 
                user_id, 
                use_rag=True,
                domain=domain
            )
            
            email_response = {
                "id": email.get("id", ""), "email_id": email.get("id"),
                "thread_id": email.get("thread_id"), "sender": email.get("sender"),
                "subject": email.get("subject"),
                "body": email.get("body", "")[:200] + "..." if len(email.get("body", "")) > 200 else email.get("body"),
                "rag_context_used": result.get("rag_context", ""),
                "model": result.get("model", "")
            }

            if result.get("no_answer_needed") or result.get("reason_category") in ["NEWSLETTER", "NOTIFICATION"]:
                email_response.update({"decision": "NO_RESPONSE_NEEDED", "reason_category": result.get("reason_category", "UNKNOWN"), "reason": result.get("reason_text", "")})
                filtered_emails.append(email_response)
            elif result.get("reason_category") in ["INSUFFICIENT_CONTEXT", "OUT_OF_SCOPE"]:
                email_response.update({"decision": "NO_REPLY_POSSIBLE", "reason_category": result.get("reason_category"), "reason": result.get("reason_text", "")})
                filtered_emails.append(email_response)
            else:
                email_response.update({"decision": "DRAFT_NEEDED", "reason_category": "DRAFT_NEEDED"})
                draft_needed_emails.append(email_response)
        
        return {
            "responses": filtered_emails + draft_needed_emails,
            "filtered_emails": filtered_emails,
            "draft_needed_emails": draft_needed_emails,
            "total_processed": len(email_list)
        }

