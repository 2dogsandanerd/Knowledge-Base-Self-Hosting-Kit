"""
Onboarding service for the Developer Edition.
Handles the initial setup and introduction for new users.
"""
from typing import Dict, Any
from src.core.feature_limits import FeatureLimits, Edition
from loguru import logger


class OnboardingService:
    """
    Handles onboarding for new users of the Developer Edition.
    """
    
    def __init__(self, edition: Edition = Edition.DEVELOPER):
        self.edition = edition
    
    def get_welcome_message(self) -> Dict[str, Any]:
        """
        Get the welcome message for the current edition.
        """
        limits = FeatureLimits.get_limits(self.edition)
        
        welcome_msg = {
            "title": "Willkommen bei Mail Modul Alpha - Developer Edition",
            "subtitle": "Ihre persönliche Wissensdatenbank",
            "edition": self.edition.value,
            "features": {
                "collections_limit": limits["max_collections"],
                "documents_per_collection": limits["max_documents_per_collection"],
                "file_formats": limits["allowed_file_formats"],
                "advanced_rag": limits["enable_advanced_rag"],
                "learning_system": limits["enable_learning_system"],
                "team_sharing": limits["enable_team_sharing"],
            },
            "welcome_text": self._get_welcome_text(),
            "next_steps": self._get_next_steps(),
            "limitations": self._get_limitations(),
            "upgrade_cta": self._get_upgrade_cta()
        }
        
        return welcome_msg
    
    def _get_welcome_text(self) -> str:
        """
        Get the welcome text based on the edition.
        """
        if self.edition == Edition.DEVELOPER:
            return (
                "Hallo! Willkommen bei Mail Modul Alpha - Developer Edition. "
                "Diese kostenlose Version ermöglicht es Ihnen, eine persönliche Wissensdatenbank "
                "auf Ihrem Computer aufzubauen und intelligente Antworten auf E-Mails zu erhalten. "
                "Ziehen Sie einfach einen Ordner mit Ihren Dokumenten hierher, um zu beginnen."
            )
        else:
            return (
                "Willkommen bei Mail Modul Alpha! "
                "Sie nutzen die {self.edition.value.title()} Edition mit erweiterten Funktionen."
            )
    
    def _get_next_steps(self) -> list:
        """
        Get the next steps for the user based on the edition.
        """
        if self.edition == Edition.DEVELOPER:
            return [
                {
                    "step": 1,
                    "title": "Erstellen Sie Ihre erste Wissenssammlung",
                    "description": "Klicken Sie auf 'Neue Sammlung erstellen' und vergeben Sie einen Namen für Ihre Wissenssammlung.",
                    "action": "create_collection"
                },
                {
                    "step": 2,
                    "title": "Dokumente hinzufügen",
                    "description": "Ziehen Sie einen Ordner mit PDF-Dokumenten in die App oder laden Sie einzelne Dateien hoch.",
                    "action": "add_documents"
                },
                {
                    "step": 3,
                    "title": "Erste Anfrage stellen",
                    "description": "Stellen Sie eine Frage zu Ihren Dokumenten im Chat-Feld unten.",
                    "action": "ask_question"
                },
                {
                    "step": 4,
                    "title": "Erste E-Mail generieren",
                    "description": "Verbinden Sie Ihr E-Mail-Konto und testen Sie die automatische Antwortgenerierung.",
                    "action": "connect_email"
                }
            ]
        else:
            return [
                {
                    "step": 1,
                    "title": "Erstellen Sie Ihre Wissenssammlungen",
                    "description": "Erstellen Sie mehrere Sammlungen für verschiedene Themen oder Abteilungen.",
                    "action": "create_collections"
                },
                {
                    "step": 2,
                    "title": "Dokumente hinzufügen",
                    "description": "Fügen Sie eine Vielzahl von Dokumentenformaten hinzu, einschließlich PDF, DOCX, TXT und mehr.",
                    "action": "add_documents"
                },
                {
                    "step": 3,
                    "title": "Teammitglieder einladen",
                    "description": "Laden Sie Kollegen ein und arbeiten Sie gemeinsam an Ihrer Wissensdatenbank.",
                    "action": "invite_team"
                }
            ]
    
    def _get_limitations(self) -> Dict[str, Any]:
        """
        Get the limitations for the current edition.
        """
        limits = FeatureLimits.get_limits(self.edition)
        
        limitations = {
            "collections": f"Sie können maximal {limits['max_collections']} Wissenssammlung{'en' if limits['max_collections'] != 1 else ''} erstellen.",
            "documents": f"Jede Sammlung kann maximal {limits['max_documents_per_collection']} Dokument{'e' if limits['max_documents_per_collection'] != 1 else ''} enthalten.",
            "formats": f"Sie können nur {', '.join(limits['allowed_file_formats'])} Dateien verarbeiten.",
            "advanced_features": {
                "advanced_rag": f"Fortgeschrittene RAG-Funktionen: {'Verfügbar' if limits['enable_advanced_rag'] else 'Nicht verfügbar (Upgrade erforderlich)'}",
                "learning_system": f"Lernsystem: {'Verfügbar' if limits['enable_learning_system'] else 'Nicht verfügbar (Upgrade erforderlich)'}",
                "team_sharing": f"Team-Sharing: {'Verfügbar' if limits['enable_team_sharing'] else 'Nicht verfügbar (Upgrade erforderlich)'}"
            }
        }
        
        return limitations
    
    def _get_upgrade_cta(self) -> Dict[str, str]:
        """
        Get the upgrade call-to-action for the current edition.
        """
        if self.edition == Edition.DEVELOPER:
            return {
                "title": "Möchten Sie mehr Funktionen?",
                "description": "Upgrade auf die Team Edition für unbegrenzte Sammlungen, erweiterte RAG-Funktionen und Team-Zugriff.",
                "button_text": "Jetzt zur Team Edition",
                "link": "/upgrade"
            }
        else:
            return {
                "title": "Alles unter Kontrolle!",
                "description": "Sie nutzen bereits eine leistungsstarke Edition mit erweiterten Funktionen.",
                "button_text": "Zur App",
                "link": "/dashboard"
            }


# Global instance for the onboarding service
onboarding_service = OnboardingService()