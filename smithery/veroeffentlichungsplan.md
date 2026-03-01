# Smithery-Veröffentlichung für Knowledge Base Self-Hosting Kit - Detaillierter Plan

## Phase 1: Vorbereitung (Vor der Veröffentlichung)

### 1.1 Code-Review & Optimierung
- [x] Überprüfe aktuelle MCP-Server-Implementierung
- [x] Überprüfe package.json im mcp-server Ordner
- [x] Analysiere Konfigurationsmöglichkeiten (config.ts)
- [ ] Ergänze fehlende Fehlerbehandlung für Offline-Szenarien
- [ ] Aktualisiere README.md im mcp-server Ordner für Smithery-Nutzer

### 1.2 Smithery-Konfiguration
- [x] Erstelle smithery.yaml mit korrekten Parametern
- [ ] Teste Konfiguration gegen Smithery-Spezifikation

### 1.3 Dokumentation
- [ ] Erstelle Installationsanleitung für Smithery-Nutzer
- [ ] Erstelle Problembehebungsabschnitt für häufige Fehler

## Phase 2: Test & Validierung

### 2.1 Lokaler Test
- [ ] Teste MCP-Server unabhängig vom Backend
- [ ] Teste Verbindung mit laufendem Knowledge Base Self-Hosting Kit Backend
- [ ] Teste Fehlerfälle (Backend nicht erreichbar)

### 2.2 Smithery-Kompatibilität
- [ ] Validiere smithery.yaml gegen Schema
- [ ] Teste npx-Aufruf des veröffentlichten Pakets

## Phase 3: Veröffentlichung

### 3.1 NPM-Publikation
- [ ] Stelle sicher, dass @knowledge-kit/mcp-server auf NPM veröffentlicht ist (aktuell: 1.0.0)
- [ ] Überprüfe Zugriffsberechtigungen für das Paket
- [ ] Stelle sicher, dass das Paket öffentlich sichtbar ist

### 3.2 Smithery-Registrierung
- [ ] Gehe zu smithery.ai/register
- [ ] Reiche das GitHub-Repo ein
- [ ] Stelle sicher, dass smithery.yaml korrekt erkannt wird

## Phase 4: Nachverfolgung & Support

### 4.1 Monitoring
- [ ] Beobachte Installationszahlen
- [ ] Sammle Nutzerfeedback

### 4.2 Support-Dokumentation
- [ ] Erstelle FAQ für häufige Fragen
- [ ] Bereite Support-Vorlagen vor

---

## Detaillierte Schritte für jede Phase

### Phase 1: Vorbereitung

#### 1.1 Code-Review & Optimierung

Aktuelle Situation:
- MCP-Server ist bereits implementiert und funktioniert
- Package.json ist korrekt konfiguriert mit:
  - Name: @knowledge-kit/mcp-server
  - Bin-Eintrag: knowledge-kit-mcp -> ./build/server.js
  - Abhängigkeiten korrekt definiert
- Konfiguration erfolgt über Umgebungsvariablen (KNOWLEDGE_BASE_API_URL, KNOWLEDGE_BASE_TIMEOUT, LOG_LEVEL)

Zu ergänzen:
- Bessere Fehlermeldung wenn Backend nicht erreichbar ist
- Hinweis in der Ausgabe dass Knowledge Base Self-Hosting Kit Backend gestartet sein muss

#### 1.2 Smithery-Konfiguration

Die smithery.yaml wurde erstellt mit:
- Startbefehl: npx -y @knowledge-kit/mcp-server
- Unterstützt stdio-Transport (korrekt für MCP)
- Definiert Umgebungsvariablen mit Beschreibungen und Standardwerten

#### 1.3 Dokumentation

Aktualisiere mcp-server/README.md mit:
- Smithery-spezifischen Installationsanweisungen
- Klarem Hinweis dass Knowledge Base Self-Hosting Kit Backend separat laufen muss
- Häufige Fehler und Lösungen

### Phase 2: Test & Validierung

Teste folgende Szenarien:
1. MCP-Server funktioniert wenn Backend läuft
2. MCP-Server gibt sinnvolle Fehlermeldung wenn Backend nicht erreichbar
3. Umgebungsvariablen werden korrekt verarbeitet
4. Smithery-Konfiguration funktioniert wie erwartet

### Phase 3: Veröffentlichung

Da das Paket @knowledge-kit/mcp-server bereits in der package.json vorhanden ist, ist es wahrscheinlich bereits auf NPM veröffentlicht. Falls nicht, müssen folgende Schritte durchgeführt werden:
- npm login (mit entsprechenden Berechtigungen)
- cd mcp-server && npm publish --access public

Anschließend Registrierung bei Smithery:
- Gehe zu https://smithery.ai/register
- Gib GitHub-Repo-URL an
- Smithery findet die smithery.yaml und konfiguriert das Tool

### Phase 4: Nachverfolgung

Nach der Veröffentlichung ist wichtig:
- Feedback von Nutzern zu sammeln
- Installationszahlen zu beobachten
- Eventuelle Probleme zeitnah zu beheben
