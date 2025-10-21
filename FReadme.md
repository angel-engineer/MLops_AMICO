## 00-Setup: Puesta a punto de entorno y flujo de trabajo

Esta carpeta es tu punto de partida para el curso. Te ayuda a preparar un entorno de desarrollo fiable y reproducible con Git/GitHub y Python utilizando herramientas modernas de dependencias.

Puedes usar uv (rápido y sencillo) o Poetry (más completo). En este curso recomendamos uv por su rapidez y excelentes instalaciones reproducibles, pero también incluimos instrucciones para Poetry.

### Inicio rápido (recomendado: uv)

- macOS (zsh/bash):

```bash
# 1) Install Git
brew install git

# 2) Install Python 3.11+ (recommended 3.11 or 3.12)
brew install python@3.11

# 3) Install uv
brew install uv
exec "$SHELL" # reload shell so 'uv' is on PATH

# 4) Clone the course repo (replace with your fork URL if contributing)
git clone <your-repo-url>
cd <repo-name>

# 5) Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 6) (If a pyproject.toml exists) install dependencies
uv sync

# 7) Verify
python -V
pip -V
uv --version
```

- Windows (PowerShell):

```powershell
# 1) Install Git
winget install --id Git.Git -e --source winget

# 2) Install Python 3.11+
winget install --id Python.Python.3.11 -e --source winget

# 3) Install uv
irm https://astral.sh/uv/install.ps1 | iex

# 4) Clone the course repo (replace with your fork URL if contributing)
git clone <your-repo-url>
cd <repo-name>

# 5) Create and activate a virtual environment
uv venv
.\.venv\Scripts\Activate.ps1

# 6) (If a pyproject.toml exists) install dependencies
uv sync

# 7) Verify
python -V
pip -V
uv --version
```

Si prefieres Poetry, consulta `02-python-envs.md` y `03-dependency-management.md`.

### Qué aprenderás aquí

- Fundamentos de Git y GitHub: instalación, autenticación SSH, ramas y PRs
- Entornos de Python: venv básico, uv y Poetry
- Gestión de dependencias: instalación, bloqueo y reproducción de entornos
- Herramientas: configuración del editor, calidad de código (ruff/black), hooks de pre-commit
- Reproducibilidad e integración continua con GitHub Actions

### Contenidos

- `01-git-github.md`: Instalación de Git/GitHub, SSH, flujos de trabajo y PRs
- `02-python-envs.md`: Instalación de Python, entornos virtuales, uv/Poetry
- `03-dependency-management.md`: Gestión y bloqueo de dependencias con uv y Poetry
- `04-tooling.md`: VS Code, ruff, black, pre-commit
- `05-data-and-secrets.md`: Archivos .env y protección de secretos (local y CI)
- `06-github-actions.md`: CI de ejemplo para uv y Poetry
- `07-os-notes.md`: Consejos para macOS y Windows (shells, rutas, permisos)
- `templates/`: ejemplos listos para copiar (pyproject, CI, pre-commit, .env, .gitignore)
- `scripts/`: scripts de configuración multiplataforma para uv y Poetry

### Cómo usar esta carpeta

- Sigue `01-03` en orden para una configuración fluida.
- Copia los archivos que necesites desde `templates/` a la raíz del proyecto (por ejemplo, `.pre-commit-config.yaml`, `.env.example`, workflows de CI).
- Ejecuta los scripts en `scripts/` si quieres una configuración guiada.

### Si necesitas ayuda

- Revisa de nuevo las notas específicas por sistema en `07-os-notes.md`.
- Confirma el comando correcto de activación del entorno para tu sistema operativo.
- Asegúrate de estar dentro del entorno virtual antes de instalar o ejecutar nada.
