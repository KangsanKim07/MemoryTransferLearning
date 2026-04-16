import litellm
import time

# 사용할 모델 이름 (ollama list에서 본 이름과 동일하게)
model_name = "ollama/qwen2:7b-coder"
messages = [{"role": "user", "content": "1+1은 뭐야?"}]

print(f"'{model_name}' 모델에 연결을 시도합니다...")

try:
    start_time = time.time()
    
    response = litellm.completion(
        model=model_name,
        messages=messages
    )
    
    end_time = time.time()
    
    print("\n✅ 성공! 응답을 받았습니다.")
    print(f"응답 시간: {end_time - start_time:.2f} 초")
    print("\n[모델 응답]")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\n❌ 실패! 오류가 발생했습니다: {e}")