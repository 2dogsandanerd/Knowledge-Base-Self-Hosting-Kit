# MCP-basierte Ingestion für Knowledge Base Self-Hosting Kit

## Übersicht

Derzeit unterstützt der MCP-Server **keine direkte Ingestion** von Dokumenten über OpenClaw. Die Ingestion erfolgt über die Knowledge Base Self-Hosting Kit-REST-API, die über die Web-Oberfläche oder direkte API-Aufrufe genutzt werden kann.

## Aktuelle Architektur

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenClaw      │    │   MCP Server    │    │   Knowledge Base Self-Hosting Kit       │
│   (WhatsApp/    │◄──►│   (stdio)       │◄──►│   (REST API)    │
│   Telegram)     │    │   @knowledge-kit/     │    │                 │
│                 │    │   mcp-server    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                                                    ▼
                                            ┌─────────────────┐
                                            │   ChromaDB      │
                                            │   (Vector DB)   │
                                            └─────────────────┘
```

## Verfügbare MCP-Tools

### 1. `query_knowledge` - Wissensabfrage
```bash
# In OpenClaw
query_knowledge(query="Was steht im Vertrag?", collections=["vertraege"], k=5)
```
*Tipp: Wenn Sie `collections` leer lassen oder weglassen, werden automatisch alle verfügbaren Sammlungen durchsucht.*

### 2. `list_collections` - Verfügbare Sammlungen
```bash
# In OpenClaw
list_collections()
```

## Ingestion-Methoden

### Methode 1: Web-Oberfläche (Empfohlen)
1. Öffnen Sie `http://localhost:8080`
2. Nutzen Sie das Upload-Formular
3. Wählen Sie Dateien aus und legen Sie die Collection fest

### Methode 2: REST-API (Direkt)
```bash
# Collection erstellen
curl -X POST http://localhost:8080/api/v1/rag/collections \
  -F "collection_name=my_collection" \
  -F "embedding_provider=ollama" \
  -F "embedding_model=nomic-embed-text"

# Dokumente hochladen
curl -X POST http://localhost:8080/api/v1/rag/documents/upload \
  -F "files=@document.pdf" \
  -F "collection_name=my_collection"
```

### Methode 3: Folder-Ingestion
```bash
# Ordner konfigurieren (in .env)
DOCS_DIR=/pfad/zu/ihren/dokumenten

# Ingestion starten
curl -X POST http://localhost:8080/api/v1/rag/ingest-folder \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/host_root",
    "collection_name": "meine_dokumente",
    "profile": "documents",
    "recursive": true
  }'
```

## Zukünftige Erweiterung: Ingestion über MCP

Für zukünftige Versionen ist geplant, folgende MCP-Tools zu implementieren:

### Geplantes Tool: `ingest_document`
```bash
# In OpenClaw (zukünftig)
ingest_document(
  file_path="/pfad/zum/dokument.pdf",
  collection="neue_sammlung",
  chunk_size=512
)
```

### Geplantes Tool: `create_collection`
```bash
# In OpenClaw (zukünftig)
create_collection(
  name="neue_sammlung",
  embedding_model="nomic-embed-text"
)
```

## Warum keine Ingestion über MCP in der aktuellen Version?

1. **Sicherheitsgründe**: Dateiuploads über Chat-Interfaces bergen Risiken
2. **Komplexität**: MCP-Protokoll ist besser für Abfragen geeignet
3. **Performance**: Große Dateien über Chat-Protokolle zu übertragen ist ineffizient
4. **Benutzerfreundlichkeit**: Web-Oberfläche ist für Uploads besser geeignet

## Empfohlene Arbeitsabläufe

### Für Endbenutzer:
1. Dokumente über Web-Oberfläche oder API einfügen
2. MCP-Tools für Abfragen nutzen
3. Collection-Management über API/Oberfläche

### Für Entwickler:
1. Nutzen Sie die REST-API für automatisierte Ingestion
2. Implementieren Sie eigene Skripte für Massenuploads
3. Nutzen Sie MCP für Abfragen innerhalb von Agenten

## Beispiel-Workflow

1. **Setup**:
   ```bash
   # Knowledge Base Self-Hosting Kit starten
   docker compose up -d
   
   # Dokumente einfügen
   curl -X POST http://localhost:8080/api/v1/rag/documents/upload \
     -F "files=@vertrag.pdf" \
     -F "collection_name=vertraege"
   ```

2. **Nutzung über OpenClaw**:
   ```bash
   # MCP-Server installieren
   openclaw mcp add --transport stdio knowledge-kit npx -y @knowledge-kit/mcp-server
   
   # Abfrage über WhatsApp/Telegram
   "Suche im Wissensspeicher nach: Vertragsklausel"
   ```

## Hinweise für Betreiber

- Stellen Sie sicher, dass die DOCS_DIR-Konfiguration korrekt ist für Folder-Ingestion
- Nutzen Sie die Health-Checks um den Status zu überwachen
- Beachten Sie die Netzwerkkonfiguration für Ollama-Verbindung