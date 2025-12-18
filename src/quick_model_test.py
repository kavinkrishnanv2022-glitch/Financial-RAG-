"""
Quick Model Test - Test with timeout for each model
"""
import subprocess
import json

print("=" * 80)
print("🧪 QUICK MODEL TEST WITH TIMEOUT")
print("=" * 80)

# Simple context and question
context = "Revenue from US customers: $100B. International revenue: $50B."
question = "What are the main revenue streams?"

prompt = f"""Answer based on: {context}
Question: {question}
Answer:"""

models = ["orca-mini", "neural-chat", "mistral"]

for model in models:
    print(f"\n{'='*80}")
    print(f"Testing: {model}")
    print(f"{'='*80}\n")
    
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.3,
        "num_predict": 100
    }
    
    # Call via curl (more reliable)
    cmd = [
        "curl",
        "-X", "POST",
        "http://localhost:11434/api/generate",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(body),
        "--max-time", "120"
    ]
    
    try:
        print(f"⏳ Calling {model}... (timeout: 120 seconds)")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                answer = response.get("response", "").strip()
                
                print(f"✅ {model} Response:")
                print("-" * 80)
                print(answer)
                print("-" * 80)
                
                # Analysis
                if len(answer) < 20:
                    print(f"⚠️  Very short response")
                elif len(answer) > 300:
                    print(f"⚠️  Long response ({len(answer)} chars)")
                else:
                    print(f"✅ Good length ({len(answer)} chars)")
                    
            except json.JSONDecodeError:
                print(f"❌ Could not parse response")
                print(result.stdout[:200])
        else:
            print(f"❌ Error: {result.stderr[:200]}")
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {model} took >130 seconds")
    except FileNotFoundError:
        print(f"❌ curl not found - trying with PowerShell...")

print("\n" + "=" * 80)
