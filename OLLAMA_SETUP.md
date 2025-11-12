# 🏠 Using Ollama (Local LLM - FREE!)

## Why Ollama?

✅ **100% FREE** - No API costs
✅ **Private** - Your data never leaves your machine
✅ **Fast** - No network latency (after model loads)
✅ **Offline** - Works without internet (after setup)
✅ **No Quotas** - Analyze unlimited videos

---

## Quick Start

### 1. Install Ollama (Already Done! ✅)

Ollama has been installed via Homebrew and is running.

### 2. Download Model (In Progress...)

The recommended model `qwen2.5:7b` is currently downloading.

You can check progress with:
```bash
ollama list
```

### 3. Ready to Use!

Once the model finishes downloading, your agent is ready to use **for free**!

```bash
python main.py
```

---

## Recommended Models

### **qwen2.5:7b** (Default - Best for this use case)
- **Size**: 4.7 GB
- **Quality**: Excellent at structured JSON output
- **Speed**: Fast on Apple Silicon
- **Best for**: Stock analysis, structured data extraction

### Alternatives

**llama3.1:8b**
```yaml
ollama_model: llama3.1:8b
```
- Size: 4.7 GB
- Great general purpose model
- Good reasoning capabilities

**mistral:7b**
```yaml
ollama_model: mistral:7b
```
- Size: 4.1 GB
- Faster than Llama
- Good for quick analysis

**For better quality (needs more RAM):**

**qwen2.5:14b**
```yaml
ollama_model: qwen2.5:14b
```
- Size: 9 GB
- Better reasoning
- Requires 16GB+ RAM

---

## Configuration

### Using Ollama (Default)

```yaml
# config.yaml
analysis_mode: ollama
ollama_model: qwen2.5:7b
ollama_base_url: http://localhost:11434
```

### Auto Mode (Tries Ollama First)

```yaml
# config.yaml
analysis_mode: auto  # Tries Ollama, falls back to cloud
```

This way you use Ollama when available, but can still use cloud providers as backup.

---

## How It Works

1. **Fetch Transcript**: Gets text from YouTube (same as other modes)
2. **Send to Ollama**: Processes locally on your machine
3. **Structured Output**: Ollama enforces JSON schema for reliable parsing
4. **Extract Stocks**: Identifies tickers, sentiment, reasoning
5. **Zero Cost**: All processing happens locally!

---

## Performance

### Speed (Apple Silicon M1/M2/M3)

| Model | First Load | Per Video | Quality |
|-------|------------|-----------|---------|
| qwen2.5:7b | ~10s | ~5-15s | ⭐⭐⭐⭐⭐ |
| llama3.1:8b | ~12s | ~8-20s | ⭐⭐⭐⭐ |
| mistral:7b | ~8s | ~4-10s | ⭐⭐⭐⭐ |

**Note**: First load is slower as model loads into RAM. Subsequent analyses are fast!

### Memory Usage

- **qwen2.5:7b**: ~6 GB RAM
- **llama3.1:8b**: ~6 GB RAM
- **qwen2.5:14b**: ~10 GB RAM

Recommended: 16GB RAM for smooth operation

---

## Cost Comparison

### Ollama (Local)
```
Cost per video: $0.00
Cost per 100 videos: $0.00
Cost per month (daily runs): $0.00

💰 Total: FREE
```

### Gemini
```
Cost per video: ~$0.02
Cost per 100 videos: ~$2.00
Cost per month (daily, 3 channels, 5 videos): ~$9.00
```

### OpenAI (GPT-4)
```
Cost per video: ~$0.05
Cost per 100 videos: ~$5.00
Cost per month: ~$22.50
```

**Savings**: $9-22/month with Ollama! 🎉

---

## Troubleshooting

### "Failed to connect to Ollama"

Make sure Ollama is running:
```bash
brew services start ollama

# Or run manually
ollama serve
```

### "Model not found"

Download the model:
```bash
ollama pull qwen2.5:7b
```

Check available models:
```bash
ollama list
```

### Slow Performance

1. **Close other apps** - Free up RAM
2. **Use smaller model** - Try `mistral:7b`
3. **Check CPU usage** - Ollama uses CPU/GPU

### Out of Memory

Use a smaller model:
```yaml
ollama_model: qwen2.5:3b  # Smaller, uses less RAM
```

Or close other applications.

---

## Advanced Usage

### Custom Ollama Server

If running Ollama on another machine:
```yaml
ollama_base_url: http://192.168.1.100:11434
```

### Model Parameters

You can tune model behavior in `src/analyzers/ollama_analyzer.py`:
```python
options={
    'temperature': 0.1,  # Lower = more deterministic
    'num_predict': 2000,  # Max output tokens
    'top_p': 0.9,        # Nucleus sampling
}
```

### GPU Acceleration

Ollama automatically uses Metal (Apple Silicon) or CUDA (NVIDIA) if available.

Check GPU usage:
```bash
# macOS
sudo powermetrics --samplers gpu_power -i1000

# Linux with NVIDIA
nvidia-smi
```

---

## Switching Between Providers

You can easily switch between Ollama and cloud providers:

### Use Ollama for Most, Cloud for Important
```yaml
analysis_mode: ollama  # Daily runs
```

When you need best quality (visual analysis):
```yaml
analysis_mode: gemini  # Special analysis
```

### Hybrid Approach
```bash
# Run daily with Ollama (free)
python main.py --config config-ollama.yaml

# Run weekly with Gemini (better quality)
python main.py --config config-gemini.yaml
```

---

## Model Management

### List Models
```bash
ollama list
```

### Pull New Model
```bash
ollama pull llama3.1:8b
```

### Remove Old Model
```bash
ollama rm mistral:7b
```

### Check Model Info
```bash
ollama show qwen2.5:7b
```

---

## Tips for Best Results

1. **Start with qwen2.5:7b** - Best balance of quality and speed
2. **Use auto mode** - Tries Ollama first, cloud as backup
3. **Keep Ollama running** - Faster startup for analysis
4. **Monitor first run** - Model downloads on first use
5. **Adjust temperature** - Lower (0.1) for consistent output

---

## Benefits Summary

| Feature | Ollama | Cloud APIs |
|---------|--------|------------|
| Cost | FREE | $9-22/month |
| Privacy | 100% local | Data sent to cloud |
| Speed | Fast (after load) | Fast |
| Offline | ✅ Yes | ❌ No |
| Setup | Medium | Easy |
| Quality | Very Good | Excellent |
| Quotas | None | Yes |

---

## Next Steps

1. ✅ Ollama installed
2. ⏳ Model downloading (qwen2.5:7b)
3. 🚀 Ready to use once complete!

**Check download progress:**
```bash
ollama list
```

**Once complete, run:**
```bash
python main.py
```

Enjoy unlimited, free stock analysis! 🎉
