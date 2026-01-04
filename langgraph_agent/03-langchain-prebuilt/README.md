# [Example] LangChain 내장 도구 실습 (생성·호출·실행) #3 학습 내역
- 브랜치명 : feature/1-langchain-toolcalling-agent/3-langchain-prebuilt
- 학습 목표 : Tool Calling의 개념 이해 및 angChain 내장 도구(Tavaily 웹 검색 등)를 직접 생성하고, 호출 및 실행 과정을 실습

# Tool Calling(도구 호출)

### **1. 개념**

> **LLM**이 외부 기능이나 데이터에 접근할 수 있게 해주는 매커니즘
> 
> 
> → **LLM의 한계를 극복하는 방법 (최신 정보 부족, 특정 작업 수행 불가 등) ⇒ `RAG` 에서 중요**
> 
> ---
> 

**외부 도구 연동의 필요성**

- 실시간 데이터 접근(시간)
- 특수 기능 수행
- 정확성 향상

---

### 그래서 Tool Calling이란?

사람이 표현한 자연어를 **API를 호출하여 사용할 수 있도록** **LLM이 “구조화된 자료 형태 - Schema”로 가공하는 것**

→ LLM이 사용할 수 있는 도구를 `Bind` 해줌

---

<aside>

**배경**

---

**Input**

- `“What is 2 times 3”`

**LLM이 가진 Tool**

- **”multiply란 이름의 곱하기를 수행하는 함수, 인자는 a, b**

---

**LLM이 Tool이 사용하는 Schema에 맞게 구조화**

- arguments: `{"a":2, "b":3}`
- name: `multiply`

**이후 Tool 호출**

</aside>

---

## 2. LangChain에서 제공하는 내장 도구(Tool)

> `검색`, `코드 인터프리터`, `생산성 도구` 등 다양한 도구를 직접/제휴 형태로 제공
> 
> 
> ex - 검색: DuckDuckGo, TavilySearch 등
> 

### 1) Tool의 구성 요소

> `name`: 도구 이름
> 
> 
> `description`: 도구가 수행하는 작업에 대한 설명
> 
> `JSON schema`: 도구의 입력을 정의하는 스키마
> 
> `function`: 실행할 함수 ( 선택적으로 비동기 함수도 가능 )
> 
- **LLM**은 `name`과 `description`을 통해 도구가 어떤 역할을 하는지 파악하기 때문에 굉장히 중요한 요소임 (프롬프트 내부에 포함되는 요소)

### Tavily Search의 도구 속성

<aside>

**Tavily 웹 검색 도구**

- AI 기반의 웹 검색 API를 제공하는 서비스
- 인증키 : 환경 변수 **TAVILY_API_KEY** 설정
- 월 1,000 콜 무료
</aside>

```python
# 도구 속성
print("자료형: ")
print(type(web_search))
print("-"*100)

print("name: ")
print(web_search.name)
print("-"*100)

print("description: ")
pprint(web_search.description)
print("-"*100)

print("schema: ")
pprint(web_search.args_schema.schema())
print("-"*100)
```

```
자료형: 
<class 'langchain_community.tools.tavily_search.tool.TavilySearchResults'>
----------------------------------------------------------------------------------------------------
name: 
tavily_search_results_json
----------------------------------------------------------------------------------------------------
description: 
('A search engine optimized for comprehensive, accurate, and trusted results. ' 
 'Useful for when you need to answer questions about current events. Input ' 
 'should be a search query.')
----------------------------------------------------------------------------------------------------
schema: 
{'description': 'Input for the Tavily tool.',
 'properties': {'query': {'description': 'search query to look up',
                          'title': 'Query',
                          'type': 'string'}},
 'required': ['query'],
 'title': 'TavilyInput',
 'type': 'object'}
----------------------------------------------------------------------------------------------------
```

### 2) Tool 호출

> LLM을 통한 Tool Calling을 사용하면 사용자의 질의를 가지고 그대로 검색하는 것이 아니라
> 
> 
> 적절한 검색어로 변환하는 과정을 수행한 후 도구를 실행한다.
> 

### Tool 직접 실행

```python
from langchain_community.tools import TavilySearchResults

# 검색할 쿼리 설정
query = "스테이크와 어울리는 와인을 추천해주세요."

# Tavily 검색 도구 초기화 (최대 2개의 결과 반환)
web_search = TavilySearchResults(max_results=2)

# 웹 검색 실행
search_results = web_search.invoke(query)

# 검색 결과 출력
for result in search_results:
    print(result)  
    print("-" * 100)  
=================================================
{'url': 'https://m.blog.naver.com/wineislikeacat/223096696241', 'content': '카베르네 소비뇽(Carbernet Sauvignon), 시라(Syrah) 품종을 추천드려요!

​

안심 스테이크와 어울리는 와인

> 안심 스테이크와 어울리는 와인 품종은? 산지오베제!

고기 본연의 맛을 즐기기 가장 좋은 부위로 꼽히는 안심은

등심 안쪽에 위치해 있어 운동량이 적기 때문에 소고기 중 육질이 가장 부드럽습니다.

지방이 거의 없기 때문에 고기 자체의 맛을 가장 잘 느낄 수 있는 것이죠.

​

이와 어울리는 품종은 산도가 높은 편에 속해 시큼한 맛으로 안심의 감칠맛을 더해주는

이탈리아의 산지오베제(Sangiovege)를 추천드릴 수 있습니다.

​

갈비살과 어울리는 와인

> 갈비살과 어울리는 와인 품종은? 말벡, 카베르네 소비뇽!

갈비뼈에 붙어있는 살인 갈비살은

뼈에서 나오는 골즙이 육즙과 어우러져 풍미가 좋죠.

등심과 비슷한 맛이지만 더 질긴 편에 속합니다.

​

오래 씹을수록 기름기가 더 느껴지는 갈비살의 경우 [...] 본문 바로가기

# 블로그

## 카테고리 이동 나의 와인 아지트, 우리동네내와인

검색

와인별 음식 추천

등심 스테이크에는 이 와인 드셔보세요! 소고기 부위별 레드와인 추천 모음

2023. 5. 8. 21:18

이웃추가

 본문 폰트 크기 조정 가
 공유하기
 URL복사
 신고하기

우리동네내와인의 소고기 스테이크 부위별 레드와인 추천!

안녕하세요, 우리동네내와인입니다!

​

흔히 육류 스테이크 하면 레드와인이라고 알려져있죠?

이번 시간에는 이 공식을 조금 더 자세히 살펴보려 합니다.

특정 부위에 더 잘 어울리는 와인을 소개하는 방식으로요 :)

​

오늘 글을 읽으시다가 모르는 레드와인 종류가 나오면

아래 글도 한번 참고해보시길 바랍니다.

​

말벡? 쉬라즈? 그게 뭐야? 레드와인 포도 품종 알아보기! - 1편

안녕하세요, 우리동네내와인입니다 :) 오늘 가져온 와인 상식은 바로 레드와인의 원료가 되는 포도 품종입...

blog.naver.com [...] {"title":"등심 스테이크에는 이 와인 드셔보세요! 소고기 부위별 레드와인 추천 모음","source": " 와인 ..","domainIdOrBlogId":"wineislikeacat","nicknameOrBlogId":"우리동네내와인","logNo":223096696241,"smartEditorVersion":4,"outsideDisplay":false,"blogDisplay":false,"cafeDisplay":false,"lineDisplay":true,"meDisplay":true}

닫기

이 블로그 홈

우리동네내와인(wineislikeacat) 님을 이웃추가하고 새글을 받아보세요

취소 이웃추가'}
----------------------------------------------------------------------------------------------------
{'url': 'https://www.wine21.com/11_news/news_view.html?Idx=19051', 'content': '예를 들어 지방이 많은 스테이크는 풍미가 강하니 맛이 묵직한 와인을, 지방이 적은 스테이크는 비교적 가벼운 와인을 매칭하는 것이 바람직하다고 말한다'}
----------------------------------------------------------------------------------------------------
```

### Tool Calling (도구 호출)

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

# Tavily 검색 도구 초기화 (최대 2개의 결과 반환)
web_search = TavilySearchResults(max_results=2)

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 쿼리를 LLM에 전달하여 결과 얻기
# LLM도 LangChain에서 실행 가능한 runnable 객체니까 실행
ai_msg = llm_with_tools.invoke(query)
```

### Tool Calling이 필요 없는 질의의 경우

```python
# 도구 호출이 필요 없는 LLM 호출을 수행
query = "안녕하세요."
ai_msg = llm_with_tools.invoke(query)

# LLM의 전체 출력 결과 출력
pprint(ai_msg)
print("-" * 100)

# 메시지 content 속성 (텍스트 출력)
pprint(ai_msg.content)
print("-" * 100)

# LLM이 호출한 도구 정보 출력
pprint(ai_msg.tool_calls)
print("-" * 100)
=================================================
AIMessage(content='안녕하세요! 어떻게 도와드릴까요?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 82, 'total_tokens': 93, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f85bea6784', 'finish_reason': 'stop', 'logprobs': None}, id='run-c9ccda7b-388e-441d-9f29-74763ee7da54-0', usage_metadata={'input_tokens': 82, 'output_tokens': 11, 'total_tokens': 93})
----------------------------------------------------------------------------------------------------
'안녕하세요! 어떻게 도와드릴까요?'
----------------------------------------------------------------------------------------------------
[]
----------------------------------------------------------------------------------------------------
```

### Tool Calling이 필요한 질의의 경우

> `content`가 비어있다
> 
> - LLM이 텍스트 출력은 따로 하지 않았다
> 
> `tool_calls`가 2개다
> 
> - TavilySearch 자체 검색 범위가 글로벌이라
>     
>     LLM이 영어까지 생성하기로 판단한듯
>     

```python
# 도구 호출이 필요한 LLM 호출을 수행
query = "스테이크와 어울리는 와인을 추천해주세요."
ai_msg = llm_with_tools.invoke(query)

# LLM의 전체 출력 결과 출력
pprint(ai_msg)
print("-" * 100)

# 메시지 content 속성 (텍스트 출력)
pprint(ai_msg.content)
print("-" * 100)

# LLM이 호출한 도구 정보 출력
pprint(ai_msg.tool_calls)
print("-" * 100)
=================================================
AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_yj3UhwPBxVhnK6qa49kXLHqB', 'function': {'arguments': '{"query":"스테이크와 어울리는 와인"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 91, 'total_tokens': 117, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f85bea6784', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-9ed27b47-8e89-48cf-b0ab-dbb41c8f080c-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': '스테이크와 어울리는 와인'}, 'id': 'call_yj3UhwPBxVhnK6qa49kXLHqB', 'type': 'tool_call'}], usage_metadata={'input_tokens': 91, 'output_tokens': 26, 'total_tokens': 117})
----------------------------------------------------------------------------------------------------
''
----------------------------------------------------------------------------------------------------

[{'args': {'query': '스테이크와 어울리는 와인'},
  'id': 'call_yj3UhwPBxVhnK6qa49kXLHqB',
  'name': 'tavily_search_results_json',
  'type': 'tool_call'},
  {'args': {'query': 'steak wine pairing recommendations'},
  'id': 'call_NR2GKJ1239gkgasmglab0Rn',
  'name': 'tavily_search_results_json',
  'type': 'tool_call'}]
----------------------------------------------------------------------------------------------------
```

### 3. Tool Calling을 통한 도구 실행

1. **`args 스키마` 사용**
    
> `tool_call` 객체에 저장된 argument 속성 값을
> 직접 도구에 전달해서 invoke를 통해 실행
> 
> 
> ---
> 
> 예제로 치면 `query`
> 
    
```python
tool_call = ai_msg.tool_calls[0]
tool_output
 = web_search.invoke(tool_call["args"])
```
    
2. **`tool_call` 사용**
    
> `tool_call` 객체 자체를 전달해서 검색
> 
    
```python
tool_message = web_search.invoke(tool_call)
```
    
3. **`ToolMessage` 정의**
    
> `ToolMessage`라는 클래스를 이용하여
> 
> 
> 도구 호출 결과를 구조화하여 사용
> 
    
```python
from langchain_core.messages import ToolMessage
tool_message = ToolMessage(
    	content=tool_output,
    	tool_call_id=tool_call["id"],
    	name=tool_call["name"]
)
```
    
4. `Batch`실행
    
> 도구 호출이 여러 개인 경우
> 
> 
> ex - 위의 예제와 같이 `tool_calls`이 2개 이상인 경우 ( 한글 검색, 영어 검색 )
> 
    
```python
# tool_messages = web_search.batch([tool_call])
tool_messages = web_search.batch(ai_msg.tool_calls)
```
