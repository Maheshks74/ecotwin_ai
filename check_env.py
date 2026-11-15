import sys

mods = [
    'simpy', 'gymnasium', 'stable_baselines3', 'torch', 'pandas', 'numpy',
    'yaml', 'streamlit', 'plotly', 'matplotlib', 'pytest', 'requests',
    'dotenv', 'pyarrow', 'tensorboard', 'google.genai'
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

missing = []
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {m} ({ver})")
    except Exception as ex:
        print(f"  [MISSING] {m}: {ex}")
        missing.append(m)

print(f"\nMissing packages: {missing}")

# Check GPU
try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}")
except Exception:
    pass

