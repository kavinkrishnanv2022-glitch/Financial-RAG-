# Using Local LLM Instead of API

Your Financial RAG has been updated to use **Ollama** - a local, free, open-source LLM. No more API rate limits!

## Setup Instructions

### 1. Install Ollama
Download and install Ollama from: https://ollama.ai

### 2. Download a Model
Open PowerShell and run:
```powershell
ollama pull mistral
```

For other models, try:
- `ollama pull llama2` - Larger, more powerful
- `ollama pull neural-chat` - Specialized for conversations
- `ollama pull orca-mini` - Lightweight alternative

### 3. Start Ollama Server
Before running the RAG app, start the Ollama server:
```powershell
ollama serve
```

This will start a local server at `http://localhost:11434`

### 4. Run Your RAG App
In a new PowerShell terminal:
```powershell
py -m streamlit run src/app.py
```

## Benefits

✅ **No API costs** - Completely free  
✅ **No rate limits** - Use as much as you want  
✅ **Completely private** - Data never leaves your computer  
✅ **Offline capable** - Works without internet (after initial model download)  
✅ **Full control** - Choose which model to run  

## Performance Notes

- **Mistral**: Fast, good quality (recommended for first-time users)
- **Llama2**: Slower but higher quality
- **Neural-Chat**: Optimized for conversations

The first run will be slower as the model loads into memory. Subsequent queries will be faster.

## Troubleshooting

**Error: "Cannot connect to Ollama"**
- Make sure Ollama server is running: `ollama serve`
- Check that it's listening on `http://localhost:11434`

**Model not found**
- Download the model: `ollama pull mistral`
- List available models: `ollama list`

**Slow responses**
- Try a smaller model: `ollama pull neural-chat` or `ollama pull orca-mini`
- Close other applications to free up RAM

## Switching Models

To use a different model, edit `app.py` and `rag_engine.py`:
Change:
```python
llm = Ollama(model="mistral", temperature=0.3)
```
To:
```python
llm = Ollama(model="llama2", temperature=0.3)
```

Then restart the app.
