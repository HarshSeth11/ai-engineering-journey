import shutil
import os

# Add this before creating vectorstore
if os.path.exists("./chroma_eval"):
    shutil.rmtree("./chroma_eval")