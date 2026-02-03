# Включение подсказок в VS Code через settings.json

Этот способ универсальный и работает одинаково на **macOS и Windows**.

---

## Шаг 1. Открыть Command Palette

- **macOS:** `Cmd + Shift + P`
- **Windows:** `Ctrl + Shift + P`

Откроется строка ввода команд.

---

## Шаг 2. Открыть settings.json

В Command Palette начни печатать:

Preferences: Open User Settings (JSON)


## Шаг 3. Включить все подсказки (IntelliSense)

Вставь внутрь `settings.json`  
(если файл не пустой — вставляй **внутрь `{}`**, через запятую):

```json
{
  "editor.quickSuggestions": {
    "other": true,
    "comments": true,
    "strings": true
  },
  "editor.suggestOnTriggerCharacters": true,
  "editor.parameterHints.enabled": true,
  "editor.inlineSuggest.enabled": true,

  "python.languageServer": "Pylance",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.autoSearchPaths": true,
  "python.analysis.completeFunctionParens": true,
  "python.analysis.typeCheckingMode": "basic"
}


Шаг 4. Перезапустить Python Language Server

Снова открой Command Palette:
	•	macOS: Cmd + Shift + P
	•	Windows: Ctrl + Shift + P

Python: Restart Language Server

# Плагины
    Python
    Pylance
    Jupyter
    Prettier
    SQLite

    
