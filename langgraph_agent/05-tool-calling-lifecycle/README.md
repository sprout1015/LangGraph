# LangChain Tool Calling 예제 - 코드 구현

이 문서는 LangChain의 Tool Calling 기능을 활용한 예제 코드를 제공합니다. LLM이 외부 도구를 호출하고 그 결과를 처리하여 답변을 생성하는 과정을 다룹니다.

## 1. Tool Calling을 통한 LLM 답변 생성

### 1.1 LLM 체인 정의

`llm_chain` 객체는 프롬프트를 초기화하고 도구 호출을 수행합니다. 이때 `@chain` 데코레이터를 통해 `user_input` 메시지로 결과값을 만들고, 이를 `placeholder`에 저장하여 사용자의 질문에 대한 답변을 진행합니다.

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableConfig, chain
from langchain_core.messages import ToolMessage # 추가

# 오늘 날짜 설정
today = datetime.today().strftime("%Y-%m-%d")

# 프롬프트 템플릿 
prompt = ChatPromptTemplate([
		# 챗봇 역할 명시 및 금일 날짜 전달 (LLM이 시점에 대한 판단 가능)
		# -> 검색 도구 활용 시 실시간 정보 전달을 위해서는 중요한 요소
    ("system", f"You are a helpful AI assistant. Today's date is {today}."),
    # 사용자 input값
    ("human", "{user_input}"),
    # 도구 메시지
    ("placeholder", "{messages}"),
])

# ChatOpenAI 모델 초기화 
llm = ChatOpenAI(model="gpt-4o-mini")

# Tavily 웹 검색 도구 초기화
web_search = TavilySearchResults(max_results=2)

# LLM에 도구를 바인딩
llm_with_tools = llm.bind_tools(tools=[web_search])

# LLM 체인 생성 (프롬프트 실행 -> 도구 호출)
llm_chain = prompt | llm_with_tools
```

### 1.2 도구 실행 체인 정의

`web_search_chain` 함수는 사용자 질문을 파싱하고, LLM 실행을 통해 검색 질의어를 도출합니다. 이 질의어를 사용하여 `web_search` 도구를 실행하고, 최종적으로 LLM을 재실행하여 도구의 반환값(placeholder)을 반영한 답변을 생성합니다.

```python
@chain
def web_search_chain(user_input: str, config: RunnableConfig):
		# 사용자 질문 파싱
    input_ = {"user_input": user_input}
	  # LLM 실행 (검색 질의어 도출)
    ai_msg = llm_chain.invoke(input_, config=config)
    print("ai_msg: \n", ai_msg)
    print("-"*100)
	  # 배치 실행을 통한 tool_msg(검색 결과) 반환
    tool_msgs = web_search.batch(ai_msg.tool_calls, config=config)
    print("tool_msgs: \n", tool_msgs)
    print("-"*100)
    # llm_chain을 다시 실행하여 placeholder 값 수정(message값)
    return llm_chain.invoke(
			    {
				    **input_, "messages": [ai_msg, *tool_msgs]
			    }, config=config
		    )
```

**체인 실행 결과 예시:**

```python
# 체인 실행
response = web_search_chain.invoke("오늘 모엣샹동 샴페인의 가격은 얼마인가요?")
```
```
ai_msg: 
 content='' additional_kwargs={'tool_calls': [{'id': 'call_Rq69dR1M4NvUAJcbyI0PAYx9', 'function': {'arguments': '{"query":"모엣샹동 샴페인 가격 2024년 10월"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 35, 'prompt_tokens': 114, 'total_tokens': 149, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f85bea6784', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-007fd1f9-bba2-4d37-9561-647da19b569c-0' tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': '모엣샹동 샴페인 가격 2024년 10월'}, 'id': 'call_Rq69dR1M4NvUAJcbyI0PAYx9', 'type': 'tool_call'}] usage_metadata={'input_tokens': 114, 'output_tokens': 35, 'total_tokens': 149}
----------------------------------------------------------------------------------------------------
tool_msgs: 
 [ToolMessage(content='[{"url": "https://dailyshot.co/m/item/4216", "content": "모엣 샹동 임페리얼 전국 가격비교하고 구매 | 데일리샷에서 모든 와인 가격 비교하고 내 주변에서 구매하기 2024 [10월 월간 돌고래] 모엣 샹동 임페리얼 모엣 샹동 하우스의 상징 \'모엣 샹동 임페리얼\'은 모엣 샹동 하우스의 상징적인 샴페인입니다. 1869년 탄생한 이 샴페인은\xa0모엣 샹동의 독보적인 스타일을 완벽하게 보여줄 수 있는 와인으로, 밝은 과실 아로마와 상쾌한 기포를 자랑합니다. 특별한 날을 더욱 빛내주는 모엣 샹동 임페리얼 \'모엣 샹동 임페리얼\'은 초록빛이 은은하게\xa0감도는 금빛 볏짚 색깔을 띱니다. 사랑받는 샴페인, 모엣 샹동 모엣 샹동(Moet & Chandon)은 세계에서 가장 큰 샴페인 하우스입니다. \'샴페인의 마법을 세상에 나눈다\'는 신조에 따라 개성 넘치는 샴페인들을 선보이는 모엣 샹동은 오늘날 세계에서 가장 사랑받는 샴페인 생산자 중 하나로 자리 잡았습니다. 4.0 스토어 보글 팬텀 샤르도네 37,000원 4.0 (1) 3.3 스토어 윈담 에스테이트, 빈 222 샤르도네 32,000원"}, {"url": "https://m.blog.naver.com/sbj5817/223342722157", "content": "모엣샹동 임페리얼 브뤼. 청량한 탄산 가득한 홈파티 샴페인. 존재하지 않는 이미지입니다. 지난 생일 친구의 내돈내산 선물로 모엣샹동 임페리얼 샴페인을 마셔보았다. 청량하면서 시원하게 터지는 탄산감이 제대로였던 시음 후기와 함께 모엣샹동 가격 안내도 ..."}]", name='tavily_search_results_json', tool_call_id='call_Rq69dR1M4NvUAJcbyI0PAYx9', artifact={'query': '모엣샹동 샴페인 가격 2024년 10월', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': '모엣 샹동 임페리얼 전국 가격비교하고 구매 | 데일리샷에서 모든 와인 가격 비교하고 내 주변에서 구매하기 2024', 'url': 'https://dailyshot.co/m/item/4216', 'content': "모엣 샹동 임페리얼 전국 가격비교하고 구매 | 데일리샷에서 모든 와인 가격 비교하고 내 주변에서 구매하기 2024 [10월 월간 돌고래] 모엣 샹동 임페리얼 모엣 샹동 하우스의 상징 '모엣 샹동 임페리얼'은 모엣 샹동 하우스의 상징적인 샴페인입니다. 1869년 탄생한 이 샴페인은\xa0모엣 샹동의 독보적인 스타일을 완벽하게 보여줄 수 있는 와인으로, 밝은 과실 아로마와 상쾌한 기포를 자랑합니다. 특별한 날을 더욱 빛내주는 모엣 샹동 임페리얼 '모엣 샹동 임페리얼'은 초록빛이 은은하게\xa0감도는 금빛 볏짚 색깔을 띱니다. 사랑받는 샴페인, 모엣 샹동 모엣 샹동(Moet & Chandon)은 세계에서 가장 큰 샴페인 하우스입니다. '샴페인의 마법을 세상에 나눈다'는 신조에 따라 개성 넘치는 샴페인들을 선보이는 모엣 샹동은 오늘날 세계에서 가장 사랑받는 샴페인 생산자 중 하나로 자리 잡았습니다. 4.0 스토어 보글 팬텀 샤르도네 37,000원 4.0 (1) 3.3 스토어 윈담 에스테이트, 빈 222 샤르도네 32,000원", 'score': 0.9992706, 'raw_content': None}, {'title': '모엣샹동 임페리얼 가격 청량함 가득 샴페인 한잔 : 네이버 블로그', 'url': 'https://m.blog.naver.com/sbj5817/223342722157', 'content': '모엣샹동 임페리얼 브뤼. 청량한 탄산 가득한 홈파티 샴페인. 존재하지 않는 이미지입니다. 지난 생일 친구의 내돈내산 선물로 모엣샹동 임페리얼 샴페인을 마셔보았다. 청량하면서 시원하게 터지는 탄산감이 제대로였던 시음 후기와 함께 모엣샹동 가격 안내도 ...', 'score': 0.966219... [truncated]}
----------------------------------------------------------------------------------------------------
('현재 모엣샹동 임페리얼 샴페인의 가격은 약 37,000원부터 시작하는 것으로 보입니다. 정확한 가격은 판매처에 따라 다를 수 있으므로, ' 
'[여기](https://dailyshot.co/m/item/4216)에서 자세한 가격비교와 구매 정보를 확인할 수 있습니다.')
```

## 2. LangChain 사용자 정의 도구 활용 - Custom Tool

> LangChain은 사용자가 직접 도구를 정의하여 사용하는 방법으로서`@tool decorator`를 제공합니다.
> 
> 
> ---
> 
> **도구 함수 작성 가이드라인**
> 
> 1. **명확한 입출력 정의**
> 2. **단일 책임 원칙 준수**
> 3. **도구 설명 작성 (””” 사이에 넣어야 먹힘 “””)
>     
>     LLM이 도구의 기능을 정확히 이해하고 사용하도록 작성
>     

**`@tool decorator`를 활용한 사용자 정의 Tool**

- 아래처럼 정의한 도구 **blog_search**의 클래스는 자동으로 `<class 'langchain_core.tools.structured.StructuredTool'>`가 됨.

```python
from langchain_core.tools import tool
from typing import List, Dict
import requests
import os
from pprint import pprint
from textwrap import dedent

# 환경 변수에서 Naver API 키 불러오기
# NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
# NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
# 주석 처리: Naver API 키가 없으므로 임시로 주석 처리합니다.
NAVER_CLIENT_ID = "YOUR_NAVER_CLIENT_ID" # 실제 사용 시 환경 변수에서 로드하거나 유효한 키로 대체
NAVER_CLIENT_SECRET = "YOUR_NAVER_CLIENT_SECRET" # 실제 사용 시 환경 변수에서 로드하거나 유효한 키로 대체

@tool
def blog_search(query: str) -> List[Dict]:
	"""네이버 블로그 API에 검색 요청을 보냅니다."""
	url = "https://openapi.naver.com/v1/search/blog.json"
	headers = {
		"X-Naver-Client-Id": NAVER_CLIENT_ID,
		"X-Naver-Client-Secret": NAVER_CLIENT_SECRET
	}
	params = {"query": query, "display": 10, "start": 1}
	try:
		response = requests.get(url, headers=headers,	params=params)
		response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
		return response.json()['items']
	except requests.exceptions.RequestException as e:
		print(f"네이버 블로그 API 호출 중 오류 발생: {e}")
		return []
	
query = "스테이크와 어울리는 와인을 추천해주세요."
# search_results = blog_search.invoke(query) # API 키가 없어 실행 결과는 생략
```

**도구 속성 예시:**

```json
name:
blog_search
-----------------------------------
description:
네이버 블로그 API에 검색 요청을 보냅니다.
-----------------------------------
args:
{'query' : {'title': 'Query', 'type' : 'string'} }
```

```python
# pprint(blog_search.args_schema.schema()) # API 키가 없어 실행 결과는 생략
```
```
{'description': '네이버 블로그 API에 검색 요청을 보냅니다.',
 'properties': {'query' : {'title': 'Query', 'type' : 'string'} },
 'required': ['query'],
 'title': 'blog_search',
 'type': 'object'}
```

## 3. Runnable 객체 도구(tool)로 변환하기

> 문자열이나 dict 입력을 받는 Runnable을 도구로 변환할 때 `as_tool` 메서드를 사용합니다.
> 

### 3.1 Document Loader를 활용한 위키피디아 검색 도구

`WikipediaLoader`를 사용하여 위키피디아 문서를 검색하고 LangChain의 `Document` 객체로 반환하는 함수를 정의합니다. 이 함수를 `RunnableLambda`로 래핑하고 `as_tool` 메서드를 통해 도구로 변환합니다.

```python
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import List
from textwrap import dedent

## 함수 정의, 출력은 LangChain의 Document 객체 활용
# WikipediaLoader를 사용하여 위키피디아 문서를 검색하는 함수 
def search_wiki(input_data: dict) -> List[Document]:
    """Search Wikipedia documents based on user input (query) and return k documents"""
    query = input_data["query"]
    k = input_data.get("k", 2)  
    wiki_loader = WikipediaLoader(query=query, load_max_docs=k, lang="ko") # 한국 위키피디아 사용을 위한 언어 선택 (기본값 en)
    wiki_docs = wiki_loader.load()
    return wiki_docs

# 도구 호출에 사용할 입력 스키마 정의 
class WikiSearchSchema(BaseModel):
    """Input schema for Wikipedia search."""
    query: str = Field(..., description="The query to search for in Wikipedia")
    k: int = Field(2, description="The number of documents to return (default is 2)")

## LangChain에서 실행가능한 Runnable 객체로 변환
# RunnableLambda 함수를 사용하여 위키피디아 문서 로더를 Runnable로 변환 
runnable = RunnableLambda(search_wiki)
# 생성한 Runnable 객체를 도구(tool)로 변환 - 이름, 설명, 입력 Schema 기재)
wiki_search = runnable.as_tool(
    name="wiki_search",
    description=dedent("""
        Use this tool when you need to search for information on Wikipedia.
        It searches for Wikipedia articles related to the user's query and returns
        a specified number of documents. This tool is useful when general knowledge
        or background information is required.
    """),
    args_schema=WikiSearchSchema
)
```

**테스트용 위키 검색 실행 결과:**

`search_wiki` 정의 시에 `load_max_docs`(최대 검색 결과)를 2개만 설정해둔 상태이기 때문에 2개만 반환됩니다.

```python
# 위키 검색 실행
query = "파스타의 유래"
wiki_results = wiki_search.invoke({"query":query})

# 검색 결과 출력
for result in wiki_results:
    print(result)  
    print("-" * 100)
```
```
page_content='피치(이탈리아어: pici, 단수: picio 피초[*])는 이탈리아의 파스타이다. ...생략

== 각주 ==' metadata={'title': '피치 (파스타)', 'summary': '피치(이탈리아어: pici, 단수: picio 피초[*])는 이탈리아의 파스타이다. 손으로 말아서 만드는 굵은 파스타의 일종으로 스파게티면이 좀 더 굵어진 것으로 보면 된다. 토스카나주의 시에나 현에서 유래했으며 몬탈치노 지역에서는 pinci라고 부른다.
반죽은 보통 밀가루나 물로만 만든다. 달걀을 첨가하는 것은 선택적이며 가정에 따라 다르다.
밀가루 반죽을 두껍고 평평하게 밀어서 편 다음 기다란 조각으로 잘라낸다. 잘라낸 조각을 두 손바닥 사이에서 말기도 하고 테이블 위에 놓고 테이블과 손바닥 사이에서 말기도 한다. 보통 연필보다 조금 더 가는 굵기로 만든다. 스파게티나 마카로니와 달리 이 파스타는 크기가 정해진 바가 없으며 길이에 따라 그 굵기도 달라진다.
먹는 경우는 여러 가지가 있지만 보통 다음 재료들을 육수나 주요 재료로 하여 요리로 만들어 먹는다.', 'source': 'https://ko.wikipedia.org/wiki/%ED%94%BC%EC%B9%98_(%ED%8C%8C%EC%8A%A4%ED%83%80)'}
----------------------------------------------------------------------------------------------------
page_content='카르보나라(이탈리아어: Carbonara)는 로마의 파스타 요리로, 계란 노른자, 경성 치즈, 염장 돼지고기, 그리고 후추를 사용해 만들 수 있다. ...생략
== 유래와 역사 ==
...생략

== 만드는 과정 ==
...생략

=== 변형 ===
...

== 참고 문헌 ==
Buccini, Anthony F. (2007). 〈On Spaghetti alla Carbonara and Related Dishes of Central and Southern Italy〉.   Hosking, Richard. 《Eggs in Cookery: Proceedings of the Oxford Symposium of Food and Cookery 2006》. Oxford Symposium. 36–47쪽. ISBN 978-1-903018-54-5.' metadata={'title': '카르보나라', 'summary': '카르보나라(이탈리아어: Carbonara)는 로마의 파스타 요리로, 계란 노른자, 경성 치즈, 염장 돼지고기, 그리고 후추를 사용해 만들 수 있다. 이 요리는 20세기 중반에 들어 현재의 형태와 명칭이 확립되었다.
치즈는 주로 페코리노 로마노를 사용하며 파르미자노 레자노나 그라나 파다노 등의 치즈를 조합할 수도 있다. 카르보나라는 주로 스파게티를 사용해서 만들지만, 페투치네, 리가토니, 링귀네, 혹은 부카티니를 사용할 수도 있다. 주로 염장 돼지고기로는 구안찰레나 판체타를 쓰지만, 이를 쉽게 구할 수 없는 해외에서는 라르돈이나 훈제 베이컨을 쓰기도 한다.', 'source': 'https://ko.wikipedia.org/wiki/%EC%B9%B4%EB%A5%B4%EB%B3%B4%EB%82%98%EB%9D%BC'}
----------------------------------------------------------------------------------------------------
```

**LLM에 도구 바인딩 후 Tool Calling 테스트:**

```python
# LLM에 도구를 바인딩 (2개의 도구 바인딩)
llm_with_tools = llm.bind_tools(tools=[web_search, wiki_search])

# 도구 호출이 필요한 LLM 호출을 수행
query = "서울 강남의 유명한 파스타 맛집은 어디인가요? 그리고 파스타의 유래를 알려주세요. "
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
```
```
AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0sdcWNlrKlEzKONnA2egk5tI', 'function': {'arguments': '{"query": "서울 강남 파스타 맛집 추천"}', 'name': 'search_web'}, 'type': 'function'}, {'id': 'call_ObK7YzEJzfjvI3wLR0nSTbzY', 'function': {'arguments': '{"query": "파스타"}', 'name': 'wiki_search'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 53, 'prompt_tokens': 150, 'total_tokens': 203, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f85bea6784', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-06b9afaf-03e4-4802-b33c-a950b765186c-0', tool_calls=[{'name': 'search_web', 'args': {'query': '서울 강남 파스타 맛집 추천'}, 'id': 'call_0sdcWNlrKlEzKONnA2egk5tI', 'type': 'tool_call'}, {'name': 'wiki_search', 'args': {'query': '파스타'}, 'id': 'call_ObK7YzEJzfjvI3wLR0nSTbzY', 'type': 'tool_call'}], usage_metadata={'input_tokens': 150, 'output_tokens': 53, 'total_tokens': 203})
----------------------------------------------------------------------------------------------------
''
----------------------------------------------------------------------------------------------------
[{'args': {'query': '서울 강남 파스타 맛집 추천'},
  'id': 'call_0sdcWNlrKlEzKONnA2egk5tI',
  'name': 'search_web',
  'type': 'tool_call'},
 {'args': {'query': '파스타'},
  'id': 'call_ObK7YzEJzfjvI3wLR0nSTbzY',
  'name': 'wiki_search',
  'type': 'tool_call'}]
----------------------------------------------------------------------------------------------------
```

**도구 실행 (위키 검색 + 요약) 결과:**

```python
ai_msg.tool_calls[1]
```
```
{'name': 'wiki_summary',
 'args': {'query': '파스타'},
 'id': 'call_ObK7YzEJzfjvI3wLR0nSTbzY',
 'type': 'tool_call'}
```
```python
tool_message = wiki_summary.invoke(ai_msg.tool_calls[1])

print(tool_message)
print("-" * 100)
pprint(tool_message.content)
```
```
# 검색 문서를 한국어로 지정했는데 프롬프트를 영어로 해서 요약을 영어로 진행한듯
content='The text discusses two main topics: pasta as a staple Italian food and the 2010 MBC drama "Pasta." \n\n1. **Pasta**: Pasta, made from durum wheat semolina mixed with water or eggs, is a key Italian food, often cooked and served in various forms. Its history dates back to ancient times, with references to similar dishes in Greek and Arabic texts. There are two main types: dried pasta (pasta secca) and fresh pasta (pasta fresca), each with distinct ingredients and preparation methods. Dried pasta is known for its durability and variety, while fresh pasta is typically made with soft wheat and eggs, often used for special dishes.\n\n2. **Drama "Pasta"**: The drama aired from January to March 2010, focusing on the journey of a young aspiring chef, Seo Yoo-kyung, as she navigates her career in an Italian restaurant and her romantic relationship with the head chef, Choi Hyun-wook. The show features various characters, including fellow chefs and restaurant staff, and received several awards for its performances. Initially planned as a 16-episode series, it was extended to 20 episodes due to its popularity.' name='wiki_summary' tool_call_id='call_ObK7YzEJzfjvI3wLR0nSTbzY'
----------------------------------------------------------------------------------------------------
('The text discusses two main topics: pasta as a staple Italian food and the ' 
'2010 MBC drama "Pasta." 
' 
'1. **Pasta**: Pasta, made from durum wheat semolina mixed with water or ' 
'eggs, is a key Italian food, often cooked and served in various forms. Its ' 
'history dates back to ancient times, with references to similar dishes in ' 
'Greek and Arabic texts. There are two main types: dried pasta (pasta secca) ' 
'and fresh pasta (pasta fresca), each with distinct ingredients and ' 
'preparation methods. Dried pasta is known for its durability and variety, ' 
'while fresh pasta is typically made with soft wheat and eggs, often used for ' 
special dishes.
' 
'2. **Drama "Pasta"**: The drama aired from January to March 2010, focusing ' 
on the journey of a young aspiring chef, Seo Yoo-kyung, as she navigates her ' 
career in an Italian restaurant and her romantic relationship with the head ' 
chef, Choi Hyun-hyun. The show features various characters, including fellow ' 
chefs and restaurant staff, and received several awards for its ' 
performances. Initially planned as a 16-episode series, it was extended to ' 
20 episodes due to its popularity.')
```

**LLM에 도구 바인딩 후 최종 답변 생성:**

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_openai import ChatOpenAI
from pprint import pprint

# 오늘 날짜 설정
today = datetime.today().strftime("%Y-%m-%d")

# 프롬프트 템플릿 
prompt = ChatPromptTemplate([
    ("system", f"You are a helpful AI assistant. Today's date is {today}."),
    ("human", "{user_input}"),
    ("placeholder", "{messages}"),
])

# LLM에 도구를 바인딩
llm_with_tools = llm.bind_tools(tools=[wiki_summary])

# LLM 체인 생성 ( 프롬프트 + 위키 요약 도구 )
llm_chain = prompt | llm_with_tools

# 도구 실행 체인 정의
@chain
def wiki_summary_chain(user_input: str, config: RunnableConfig):
    # 입력값 딕셔너리화
    input_ = {"user_input": user_input}
    # 딕셔너리로 변환한 입력값으로 도구 호출
    ai_msg = llm_chain.invoke(input_, config=config)
    print("ai_msg: \n", ai_msg)
    print("-"*100)
    # 도구 호출의 결과(ai_msg.tool_calls)를 위키 요약 도구에 전달하여 실행
    tool_msgs = wiki_summary.batch(ai_msg.tool_calls, config=config)
    print("tool_msgs: \n", tool_msgs)
    print("-"*100)
    # 최초 도구 메시지 + 위키 요약 도구 실행한 메시지를 모아서 placeholder에 전달 (prompt의 내부요소)
    # -> LLM은 사용자의 쿼리, 원문 문서, 요약 문서를 모두 prompt에서 가질 수 있다. 이를 기반으로 답변 생성(llm_chain.invoke)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)

# 체인 실행
response = wiki_summary_chain.invoke("파스타의 유래에 대해서 알려주세요.")

# 최종 답변 출력 
pprint(response.content)
```
```
ai_msg: 
 content='' additional_kwargs={'tool_calls': [{'id': 'call_Qok90XPBaLJsVPEoHtWmUfUJ', 'function': {'arguments': '{"query":"파스타의 유래"}', 'name': 'wiki_summary'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 120, 'total_tokens': 139, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f85bea6784', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-623c1f91-8d8e-4d70-bc55-095590cd4ee6-0' tool_calls=[{'name': 'wiki_summary', 'args': {'query': '파스타의 유래'}, 'id': 'call_Qok90XPBaLJsVPEoHtWmUfUJ', 'type': 'tool_call'}] usage_metadata={'input_tokens': 120, 'output_tokens': 19, 'total_tokens': 139}
----------------------------------------------------------------------------------------------------
tool_msgs: 
 [ToolMessage(content='피치(pici)는 이탈리아 토스카나주에서 유래한 손으로 만든 굵은 파스타로, 밀가루와 물로 반죽하여 길게 만들어진다. 카르보나라(carbonara)는 로마의 파스타 요리로, 계란 노른자, 경성 치즈, 염장 돼지고기, 후추를 사용하여 만든다. 이 요리는 20세기 중반에 현재의 형태로 확립되었으며, 주로 스파게티와 함께 제공된다. 카르보나라의 명칭은 여러 이론이 있으며, 석탄 광부의 식사로 유래했거나 이탈리아 통일을 주도한 비밀 결사와 관련이 있을 수 있다. 요리 방법은 파스타를 끓이고, 구안찰레를 볶은 후 계란과 치즈를 섞어 완성한다.', name='wiki_summary', tool_call_id='call_Qok90XPBaLJsVPEoHtWmUfUJ')]
----------------------------------------------------------------------------------------------------
('파스타의 유래는 이탈리아의 다양한 지역에서 발전해온 전통적인 요리입니다. 예를 들어, 피치(pici)는 토스카나주에서 유래한 손으로 만든 ' 
'굵은 파스타로, 밀가루와 물로 반죽하여 길게 만들어집니다. 
' 
'
또한, 카르보나라(carbonara)는 로마의 대표적인 파스타 요리로, 계란 노른자, 경성 치즈, 염장 돼지고기, 후추를 사용하여 ' 
'만들어집니다. 이 요리는 20세기 중반에 현재의 형태로 확립되었으며, 주로 스파게티와 함께 제공됩니다. 카르보나라의 명칭에 대해서는 여러 ' 
'이론이 있으며, 석탄 광부의 식사로 유래했거나 이탈리아 통일을 주도한 비밀 결사와 관련이 있을 수 있습니다. 
' 
'
파스타는 이처럼 지역에 따라 다양한 형태와 조리법이 존재하며, 이탈리아 요리의 중요한 부분을 차지하고 있습니다.')
```