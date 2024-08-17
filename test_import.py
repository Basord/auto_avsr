try:
    from ibug.face_alignment import FANPredictor
    print("Successfully imported FANPredictor")
except ImportError as e:
    print(f"Import failed: {e}")

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")