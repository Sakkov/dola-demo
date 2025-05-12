try:
    import transformers
    print(f"Successfully imported transformers library, version: {transformers.__version__}")
    print(f"Transformers library is installed at: {transformers.__file__}")
except ImportError:
    print("Transformers library is NOT installed or not found in the current Python environment.")
except Exception as e:
    print(f"An unexpected error occurred while trying to import transformers: {e}")

