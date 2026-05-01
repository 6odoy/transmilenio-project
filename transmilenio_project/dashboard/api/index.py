import sys
import os

# Añade el directorio dashboard/ al path para que se puedan importar
# app.py, config.py y el paquete models/ desde cualquier working directory.
_dashboard_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dashboard_dir not in sys.path:
    sys.path.insert(0, _dashboard_dir)

# Vercel espera el objeto WSGI con el nombre 'app'.
from app import server as app  # noqa: E402
