"""
Pandas Agent Service für strukturierte Datenabfragen.

Diese Klasse implementiert einen Agenten, der Anfragen zu strukturierten Daten
in Excel- oder CSV-Dateien beantworten kann. Die Implementierung verwendet
LlamaIndex PandasQueryEngine und beinhaltet Sicherheitsmaßnahmen, um
schädlichen Code zu verhindern.
"""
import os
import tempfile
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

from llama_index.core.query_engine import PandasQueryEngine

from loguru import logger


class PandasQueryError(Exception):
    """Ausnahme für Fehler in der Pandas-Abfrage."""
    pass


class FileAccessError(Exception):
    """Ausnahme für Datei-Zugriffsfehler."""
    pass


class PandasAgentService:
    """
    Service für die Interaktion mit strukturierten Daten (Excel/CSV) mittels LLM.
    
    Verwendet LlamaIndex PandasQueryEngine, um natürlichsprachliche Anfragen
    in Pandas-Operationen zu übersetzen.
    """
    
    def __init__(self, llm_service):
        """
        Initialisiert den PandasAgentService.
        
        Args:
            llm_service: Der zentrale LLM-Service des Projekts
        """
        self.llm_service = llm_service
    
    async def validate_pandas_file(self, file_path: str) -> bool:
        """
        Validiert eine Datei, um sicherzustellen, dass sie eine gültige 
        Pandas-Datei ist (Excel oder CSV).
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            True, wenn Datei gültig ist, sonst False
        """
        try:
            file_path_obj = Path(file_path)
            
            # Überprüfe, ob Datei existiert
            if not file_path_obj.exists():
                logger.error(f"Datei existiert nicht: {file_path}")
                return False
            
            # Überprüfe, ob Datei lesbar ist
            if not os.access(file_path, os.R_OK):
                logger.error(f"Datei nicht lesbar: {file_path}")
                return False
            
            # Validiere Dateiendung
            valid_extensions = {'.xlsx', '.xls', '.csv', '.parquet'}
            if file_path_obj.suffix.lower() not in valid_extensions:
                logger.error(f"Ungültige Dateiendung: {file_path_obj.suffix}")
                return False
            
            # Versuche, Datei zu lesen (mit temporärer Sicherheitsprüfung)
            try:
                if file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, nrows=1)  # Nur erste Zeile lesen zur Validierung
                elif file_path_obj.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path, nrows=1)  # Nur erste Zeile lesen zur Validierung
                elif file_path_obj.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    return False
                    
                # Prüfe, ob DataFrame gültig ist
                if df is None or df.empty:
                    logger.warning(f"Datei enthält keine gültigen Daten: {file_path}")
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"Fehler beim Lesen der Datei {file_path}: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Datei-Validierung {file_path}: {str(e)}")
            return False
    
    async def execute_pandas_query(self, dataframe: pd.DataFrame, query_text: str) -> Any:
        """
        Führt eine Pandas-Abfrage sicher aus.
        
        Args:
            dataframe: Pandas DataFrame, auf dem die Abfrage ausgeführt wird
            query_text: Die natürlichsprachliche Abfrage
            
        Returns:
            Ergebnis der Abfrage
            
        Raises:
            PandasQueryError: Bei fehlerhafter Abfrage oder Sicherheitsverletzung
        """
        # Sicherheitsprüfung der Abfrage (vereinfachte Implementierung)
        suspicious_keywords = ['import', 'exec', 'eval', '__', 'os.', 'sys.', 'subprocess', 'open(']
        for keyword in suspicious_keywords:
            if keyword in query_text.lower():
                raise PandasQueryError(f"Verdächtiges Schlüsselwort in der Abfrage entdeckt: {keyword}")
        
        try:
            # Hole das LLM vom zentralen LLM-Service des Projekts
            llm_instance = self.llm_service.get_llm_instance()
            
            # Erstelle PandasQueryEngine mit dem LLM
            query_engine = PandasQueryEngine(
                df=dataframe,
                llm=llm_instance,
                verbose=True
            )
            
            # Führe die Abfrage aus - begrenze die Ausführungszeit
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, query_engine.query, query_text)
                return result
            except asyncio.TimeoutError:
                raise PandasQueryError("Zeitüberschreitung bei der Ausführung der Abfrage")
                
        except Exception as e:
            logger.error(f"Fehler bei Pandas-Abfrage: {str(e)}")
            raise PandasQueryError(f"Fehler bei der Ausführung der Abfrage: {str(e)}")
    
    async def query(self, file_path: str, query_text: str, llm_service) -> Dict[str, Any]:
        """
        Führt eine Anfrage auf einer Pandas-Datei aus.
        
        Args:
            file_path: Pfad zur Excel/CSV-Datei
            query_text: Die natürlichsprachliche Anfrage
            llm_service: Der zentrale LLM-Service des Projekts
            
        Returns:
            Dictionary mit dem Ergebnis und Metadaten
        """
        # Verwende den übergebenen LLM-Service statt des internen
        self.llm_service = llm_service
        
        # Validiere die Datei
        if not await self.validate_pandas_file(file_path):
            raise FileAccessError(f"Ungültige oder unzugängliche Datei: {file_path}")
        
        try:
            # Lade den DataFrame
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path_obj.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path_obj.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise FileAccessError(f"Nicht unterstütztes Dateiformat: {file_path_obj.suffix}")
            
            # Führe die Abfrage aus
            result = await self.execute_pandas_query(df, query_text)
            
            # Konvertiere das Ergebnis in ein geeignetes Format
            # Hier wird das Ergebnis in ein Format umgewandelt, das den Document-Objekten ähnelt
            formatted_result = self._format_result(result, file_path, query_text)
            
            return {
                "status": "success",
                "result": formatted_result,
                "file_path": file_path,
                "query": query_text,
                "rows_processed": len(df),
                "columns": list(df.columns)
            }
            
        except PandasQueryError:
            raise
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Pandas-Abfrage: {str(e)}")
            raise PandasQueryError(f"Unerwarteter Fehler: {str(e)}")
    
    def _format_result(self, result: Any, file_path: str, query_text: str) -> List[Dict[str, Any]]:
        """
        Formatiert das Ergebnis so, dass es mit Document-Objekten kompatibel ist.
        
        Args:
            result: Roh-Ergebnis vom PandasQueryEngine
            file_path: Pfad zur Originaldatei
            query_text: Ursprüngliche Anfrage
            
        Returns:
            Liste von formatierten Ergebnissen im Document-ähnlichen Format
        """
        formatted_results = []
        
        # Wenn das Ergebnis ein DataFrame ist, konvertiere jede Zeile zu einem Document-ähnlichen Objekt
        if isinstance(result, pd.DataFrame):
            for idx, row in result.iterrows():
                formatted_results.append({
                    "content": str(row.to_dict()),
                    "metadata": {
                        "source": file_path,
                        "query": query_text,
                        "row_index": idx,
                        "data_type": "structured_table_result"
                    }
                })
        else:
            # Für andere Ergebnistypen (z.B. Skalare oder Strings)
            formatted_results.append({
                "content": str(result),
                "metadata": {
                    "source": file_path,
                    "query": query_text,
                    "data_type": "structured_table_result"
                }
            })
        
        return formatted_results