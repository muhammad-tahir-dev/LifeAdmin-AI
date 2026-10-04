import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Syntax-check every project Python file without importing external services.
for path in ROOT.rglob('*.py'):
    if any(part in {'venv', '__pycache__'} for part in path.parts):
        continue
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

main_text = (ROOT / 'main.py').read_text(encoding='utf-8')
assert 'from part5 import router as member5_router' in main_text
assert 'app.include_router(member5_router, prefix="/part5", tags=["Member 5"])' in main_text

part5_text = (ROOT / 'part5.py').read_text(encoding='utf-8')
for route in ['/document/analyze', '/finance/analyze', '/verification/check', '/adaptive-plan']:
    assert route in part5_text

print('INTEGRATION STRUCTURE TEST PASSED')
