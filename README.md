# LangChain으로 Multimodal과 RAG 활용하기 (Workshop용)

LLM (Large Language Models)을 이용한 어플리케이션을 개발할 때에 [LangChain](https://www.langchain.com/)을 이용하면 쉽고 빠르게 개발할 수 있습니다. 여기에서는 LangChain으로 Multimodal을 활용하고 RAG를 구현할 뿐아니라, Prompt engineering을 활용하여, 번역하기, 문법 오류고치기, 코드 요약하기를 구현합니다. Multimodel을 지원하는 [Anthropic Claude3](https://aws.amazon.com/ko/blogs/machine-learning/unlocking-innovation-aws-and-anthropic-push-the-boundaries-of-generative-ai-together/)는 이전 모델에서 사용하던 [LangChain Bedrock](https://python.langchain.com/docs/integrations/llms/bedrock)을 사용할 수 없고, [LangChain ChatBedrock](https://python.langchain.com/docs/integrations/chat/bedrock/)을 이용하여야 합니다. ChatBedrock은 LangChain의 [chat model component](https://python.langchain.com/docs/integrations/chat/)을 지원하며, Anthropic의 Claude 모델뿐 아니라 AI21 Labs, Cohere, Meta, Stability AI, Amazon Titan을 모두 지원합니다. 

## Architecture 개요

서버리스 Architecture를 이용하여 RAG가 적용된 Chatbot 인프라를 구성하면, 트래픽의 변화에 유연하게 대응할 수 있으며, 유지보수에 대한 부담도 줄일 수 있습니다. 주요 구현 사항은 아래와 같습니다.

- Multimodal: Text뿐 아니라 이미지를 분석할 수 있습니다. 
- RAG 구성: OpenSearch의 Vector 검색을 이용하여 빠르고 성능이 우수한 RAG를 구성할 수 있습니다.
- 인터넷 검색: RAG에 관련된 문서가 없을 경우에 Google으 Search API를 활용하여 검색된 결과를 활용합니다.
- Prority Search: RAG의 Retrieve를 이용하여 k개의 문서를 얻었지만 일부는 관련도가 낮을수 있어 정확도에 나쁜 영향을 줄 수 있습니다. Faiss의 Similarity Search로 관련된 문서(Relevant Documents)를 관련도에 따라 정렬하고 관련이 없는 문서는 제외할 수 있습니다.
- 채팅 이력의 저장 및 활용: 서버리스 서비스인 Lambda가 실행될 때에 DynamoDB에 저장된 채팅 이력을 가져와 활용합니다.
- 지속적인 대화: API Gateway를 이용하여 WebSocket을 구성하므로써 양방향 대화를 구현할 수 있습니다.
- 편리한 배포: AWS CDK로 관련된 인프라를 배포하고 편리하게 관리할 수 있습니다. 

<img src="./images/multimodal-architecture.png" width="600">

## 주요 시스템 구성

### LangChain의 ChatBedrock

Claude3부터는 ChatBedrock을 이용하여야 합니다. 이때, [LangChain ChatBedrock](https://python.langchain.com/docs/integrations/chat/bedrock/)의 max_tokens_to_sample이 ChatBedrock에서는 max_tokens로 변경되었습니다. 상세한 코드는 [lambda-chat-ws](./lambda-chat-ws/lambda_function.py)을 참조합니다. 

```python
import boto3
from langchain_aws import ChatBedrock

boto3_bedrock = boto3.client(
    service_name = 'bedrock-runtime',
    region_name = bedrock_region,
    config = Config(
        retries = {
            'max_attempts': 30
        }
    )
)

HUMAN_PROMPT = "\n\nHuman:"
parameters = {
    "max_tokens": maxOutputTokens,
    "temperature": 0.1,
    "top_k": 250,
    "top_p": 0.9,
    "stop_sequences": [HUMAN_PROMPT]
}

chat = ChatBedrock(   
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )      
```

### Chat Basic

아래는 가장 일반적인 chatbot을 위한 prompt입니다. chatbot의 이름을 지정하고 Role을 부여할 수 있습니다. Chat history는 MessagesPlaceholder()을 이용하여 반영하고, "input"에 질문을 넣은후 stream 방식으로 결과를 Client에 전달합니다. 결과의 metadata에서 사용한 토큰수에 대한 정보를 확인할 수 있습니다. 

```python
system = (
    "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
)

human = "{input}"

prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])

history = memory_chain.load_memory_variables({})["chat_history"]
        
chain = prompt | chat    
try: 
    isTyping(connectionId, requestId)  
    stream = chain.invoke(
    {
        "history": history,
        "input": query,
    }
)

msg = readStreamMsg(connectionId, requestId, stream.content)    
                    
usage = stream.response_metadata['usage']
print('prompt_tokens: ', usage['prompt_tokens'])
print('completion_tokens: ', usage['completion_tokens'])
print('total_tokens: ', usage['total_tokens'])

def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            sendMessage(connectionId, result)
    return msg
```

### 서버리스 기반으로 WebSocket 연결하기

#### Client

[Client](./html/chat.js)는 서버리스인 API Gateway를 이용하여 [WebSocket과 연결](https://docs.aws.amazon.com/ko_kr/apigateway/latest/developerguide/apigateway-WebSocket-api-overview.html)합니다. 이때 client가 연결하는 endpoint는 API Gateway 주소입니다. 아래와 같이 WebSocket을 선언한 후에 onmessage로 메시지가 들어오면, event의 'data'에서 메시지를 추출합니다. 세션을 유지하기 위해 일정간격으로 keep alive 동작을 수행합니다. 

```java
const ws = new WebSocket(endpoint);

ws.onmessage = function (event) {        
    response = JSON.parse(event.data)

    if(response.request_id) {
        addReceivedMessage(response.request_id, response.msg);
    }
};

ws.onopen = function () {
    isConnected = true;
    if(type == 'initial')
        setInterval(ping, 57000); 
};

ws.onclose = function () {
    isConnected = false;
    ws.close();
};
```

발신 메시지는 JSON 포맷으로 아래와 같이 userId, 요청시간, 메시지 타입과 메시지를 포함합니다. 발신시 [WebSocket의 send()](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/send)을 이용하여 아래와 같이 발신합니다. 만약 발신시점에 세션이 연결되어 있지 않다면 연결하고 재시도하도록 알림을 표시합니다.

```java
sendMessage({
    "user_id": userId,
    "request_id": requestId,
    "request_time": requestTime,        
    "type": "text",
    "body": message.value
})

WebSocket = connect(endpoint, 'initial');
function sendMessage(message) {
    if(!isConnected) {
        WebSocket = connect(endpoint, 'reconnect');
        
        addNotifyMessage("재연결중입니다. 잠시후 다시시도하세요.");
    }
    else {
        WebSocket.send(JSON.stringify(message));     
    }     
}
```

#### Server 

Client로 부터 메시지 수신은 Lambda로 전달된 event에서 connectionId와 routeKey를 이용해 수행합니다. 이때 keep alive 동작을 수행하여 세션을 유지합니다. 메시지 발신은 boto3로 "apigatewaymanagementapi"로 client를 정의한 후에 client.post_to_connection()로 전송합니다. 

```python
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)

def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

def lambda_handler(event, context):
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            if body[0:8] == "__ping__":  # keep alive
                sendMessage(connectionId, "__pong__")
            else:
                msg, reference = getResponse(connectionId, jsonBody) # 메시지 처리
```

### Multimodal 활용

Claude3은 Multimodal을 지원하므로 이미지에 대한 분석을 할 수 있습니다. LangChain의 ChatBedrock을 이용하여 Multimodel을 활용합니다. 이후 아래와 같이 Base64로 된 이미지를 이용해 query를 수행하면 이미지에 대한 설명을 얻을 수 있습니다. Sonnet에서 처리할 수 있는 이미지의 크기 제한으로 resize를 수행하여야 합니다. 

```python
if file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
    s3_client = boto3.client('s3') 
                    
    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
                
    width, height = img.size 
    print(f"width: {width}, height: {height}, size: {width*height}")
                
    isResized = False
    while(width*height > 5242880):                    
        width = int(width/2)
        height = int(height/2)
        isResized = True
        print(f"width: {width}, height: {height}, size: {width*height}")
                
    if isResized:
        img = img.resize((width, height))
                
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
    msg = use_multimodal(chat, img_base64, query)                       

def use_multimodal(chat, img_base64, query):    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
```

이미지에서 텍스트를 추출하는 방법은 아래와 같습니다. 추출된 텍스트를 memory chain에 저장해 놓으면, 이후 추출된 텍스트를 베이스로 답변을 얻을 수 있습니다. 

```python
text = extract_text(chat, img_base64)
extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
print('extracted_text: ', extracted_text)
if len(extracted_text)>10:
    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
memory_chain.chat_memory.add_ai_message(extracted_text)

def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text
```

### RAG를 활용하기

OpenSearch로 얻어진 관련된 문서들로 부터 Context를 얻습니다. 새로운 질문(Revised question)이 한국어/영어이면 다른 Prompt를 활용하빈다. 여기서는 <context></context> tag를 활용하여 context를 다른 문장과 구분하여 더 명확하게 LLM에게 전달할 수 있습니다. 이때, readStreamMsg()을 이용하여 얻어진 stream을 client로 전달합니다. 

```python
def query_using_RAG_context(connectionId, requestId, chat, context, revised_question):    
    if isKorean(revised_question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
                   
    chain = prompt | chat
    
    stream = chain.invoke(
        {
            "context": context,
            "input": revised_question,
        }
    )
    msg = readStreamMsg(connectionId, requestId, stream.content)    

    return msg
```

### 대화 이력의 관리

대화를 원활하게 하기 위해서는 대화이력의 관리가 필요합니다. Lambda와 같은 서비리스는 이벤트가 있을 경우에만 사용이 가능하므로 이벤트의 userId를 이용해 DynamoDB의 대화이력을 가져와서 대화에서 활용합니다.

여기서는 map_chain으로 사용자별 대화이력을 저장하는 메모리를 관리합니다. 이벤트에서 userId를 추출해서 해당 대화이력이 있는 경우에는 재활용하고, 없는 경우에는 DynamoDB에서 읽어옵니다. 대화이력이 너무 긴 경우에 context의 크기에 영향을 줄 수 있으므로, [ConversationBufferWindowMemory](https://python.langchain.com/docs/modules/memory/types/buffer_window)에서 k를 이용해 관리되는 대화이력의 크기를 조정합니다. 새로운 대화는 add_user_message()와 add_ai_message()을 이용해 추가합니다. 

```python
map_chain = dict()

if userId in map_chain:  
    memory_chain = map_chain[userId]    
else: 
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=10)
    map_chain[userId] = memory_chain

memory_chain.chat_memory.add_user_message(text)
memory_chain.chat_memory.add_ai_message(msg)
```

대화이력을 context에서 사용할 때에는 아래와 같이 읽어옵니다.

```python
history = memory_chain.load_memory_variables({})["chat_history"]
```

DynamoDB에서 대화이력은 아래와 같이 읽어옵니다.
```python
def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg)
```



### 번역하기

입력된 text가 한국어인지 확인하여 chain.invoke()의 input/output 언어 타입을 변경할 수 있습니다. 번역된 결과만을 얻기 위하여 <result></result> tag를 활용하였고, 결과에서 해당 tag를 제거하여 번역된 문장만을 추출하였습니다. 상세한 코드는 [lambda-chat-ws](./lambda-chat-ws/lambda_function.py)을 참조합니다. 

```python
def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    result = chain.invoke(
        {
            "input_language": input_language,
            "output_language": output_language,
            "text": text,
        }
    )
        
    msg = result.content
    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
```

### 문법 오류고치기

문장의 오류를 확인하는 Prompt 입니다. 상세한 코드는 [lambda-chat-ws](./lambda-chat-ws/lambda_function.py)을 참조합니다. 

```python
def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg
```

### 코드 요약하기

코드의 내용을 요약하는 Prompt 입니다. 상세한 코드는 [lambda-chat-ws](./lambda-chat-ws/lambda_function.py)을 참조합니다. 

```python
def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
```

### OpenSearch에 문서 등록하기

파일이 S3에 저장될때 발생하는 putEvent를 받아서 OpenSearch에 문서를 저장합니다. 이때, 문서가 업로드할때 생성하는 meta를 이용하여 업데이트시 삭제후 등록을 수행합니다. 상세한 코드는 [lambda-document-manager](./lambda-document-manager/lambda_function.py)을 참조합니다. 


```python
index_name = 'idx-rag'
vectorstore = OpenSearchVectorSearch(
    index_name=index_name,  
    is_aoss = False,
    #engine="faiss",  # default: nmslib
    embedding_function = bedrock_embeddings,
    opensearch_url = opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd),
)

def store_document_for_opensearch(docs, key):    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
    metadata_key = meta_prefix+objectName+'.metadata.json'
    delete_document_if_exist(metadata_key)
    
    try:        
        response = vectorstore.add_documents(docs, bulk_size = 2000)
        print('response of adding documents: ', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                

    return response

def delete_document_if_exist(metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            ids = json.loads(meta)['ids']
            result = vectorstore.delete(ids)
            print('result: ', result)        
        else:
            print('no meta file: ', metadata_key)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")
```

### OpenSearch에서 Parent/Child Chunking 활용하기

문서를 크기에 따라 parent chunk와 child chunk로 나누어서 child chunk를 찾은후에 LLM의 context에는 parent chunk를 사용하면, 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다. 상세한 내용은 [parent-document-retrieval.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/parent-document-retrieval.md)을 참조합니다.

### S3 event를 이용한 RAG 등록 및 이미지 추출

[RAG-s3-event.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/RAG-s3-event.md)에서는 S3 event를 등록하는 방법과 등록된 문서에서 이미지만 추출하는 방법을 설명합니다.

### Google Search API 활용

Google Search API를 활용하기 위해서는 [google_api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)와 [google_cse_id](https://programmablesearchengine.google.com/controlpanel/create?hl=ko)가 필요합니다. 이 값을 코드에 하드코딩하지 않기 위하여 AWS Secret Manager를 이용합니다. 아래와 같이 google_api_key와 google_cse_id을 가져옵니다. 

```python
googleApiSecret = os.environ.get('googleApiSecret')
secretsmanager = boto3.client('secretsmanager')
try:
    get_secret_value_response = secretsmanager.get_secret_value(
        SecretId=googleApiSecret
    )
    secret = json.loads(get_secret_value_response['SecretString'])
    google_api_key = secret['google_api_key']
    google_cse_id = secret['google_cse_id']

except Exception as e:
    raise e
```

OpenSearch에 검색했을때 관련된 문서가 없거나 관련도가 낮은 경우에 아래와 같이 Google Search API로 관련된 문서를 가져와서 RAG처럼 활용합니다.

```python
if len(selected_relevant_docs)==0:  # google api
    api_key = google_api_key
    cse_id = google_cse_id 
                
    relevant_docs = []
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=revised_question, cx=cse_id).execute()

    if "items" in result:
        for item in result['items']:
            api_type = "google api"
            excerpt = item['snippet']
            uri = item['link']
            title = item['title']
            confidence = ""
            assessed_score = ""
                            
            doc_info = {
                "rag_type": 'search',
                "api_type": api_type,
                "confidence": confidence,
                "metadata": {
                    "source": uri,
                    "title": title,
                    "excerpt": excerpt,
                },
                "assessed_score": assessed_score,
            }
            relevant_docs.append(doc_info)           
                
    if len(relevant_docs)>=1:
        selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embedding, minDocSimilarity)
```

### 대량의 문서 업로드시 처리 방법

다수의 RAG 문서를 S3에 업로드할때 원할한 처리를 위한 Event driven architecture입니다. RAG용 문서는 채팅 UI에서 파일업로드 버튼을 통해 업로드 할 수 있지만, S3 console 또는 AWS CLI를 이용해 S3에 직접 업로드 할 수 있습니다. 이때, OpenSearch에 문서를 업로드하는 시간보다 더 빠르게 문서가 올라오는 경우에 Queue를 통해 S3 putEvent를 관리하여야 합니다. OpenSearch에 문서 업로드시에 Embedding이 필요하므로 아래와 같이 Multi-Region의 Bedrcok Embedding을 활용합니다. 

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/7403e19b-20ca-437b-b2db-725d3c57b4f3)

### Agent 정의 및 활용

[LLM Agent](https://github.com/kyopark2014/llm-agent)와 같이, 다양한 API를 이용하기 위하여 Agent를 이용할 수 있습니다. 메뉴에서 ReAct나 ReAct chat을 이용해 기능을 확인할 수 있습니다.



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 Seoul 리전 (ap-northeast-2)을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-multimodal/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다. 

## 실행결과

### Multimodel

"Conversation Type"에서 [General Conversation]을 선택한 후에 [logo-langchain.png](./contents/logo-langchain.png) 파일을 다운로드합니다.

<img src="./contents/logo-langchain.png" width="300">


이후에 채팅창 아래의 파일 버튼을 선택하여 업로드합니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/d8252ef0-1d31-4add-9ae1-ba43bbb68dfa)

[emotion-garden.jpg](./contents/emotion-garden.jpg) 파일을 다운로드합니다.   

<img src="./contents/emotion-garden.jpg" width="300">

채팅창에 "그림속의 행사 이름만을 정확히 알려주세요. \<result\> 태그를 붙여주세요."라고 입력하고, [emotion-garden.jpg](./contents/emotion-garden.jpg)을 업로드하면 질문에 맞는 동작을 수행합니다. 이때의 결과는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-multimodal-and-rag/assets/52392004/02841097-4b67-4d2c-9342-c4c81de7848e)

[profit_table.png](./contents/profit_table.png) 파일을 다운로드합니다.  

<img src="./contents/profit_table.png" width="300">

[profit_table.png](./contents/profit_table.png)을 업로드한 후에 채팅창에 "AJ네트웍스의 PER값은?"라고 입력하면, profit_table.png에서 추출된 표로부터 아래와 같은 결과를 얻을수 있습니다.

![image](https://github.com/kyopark2014/llm-multimodal-and-rag/assets/52392004/33ccfc42-114b-426b-851b-3d54a1a6d6f8)

[claim-form.png](./contents/claim-form.png) 파일을 다운로드합니다.  

<img src="./contents/claim-form.png" width="300">

[claim-form.png](./contents/claim-form.png)를 업로드하면 아래와 같이 표를 포함한 내용에서 중요한 키워드를 추출할 수 있습니다.

![image](https://github.com/kyopark2014/llm-multimodal-and-rag/assets/52392004/88fd8317-3ecd-463a-aaf5-393236586e33)


[korean-cloth.jpeg](./contents/korean-cloth.jpeg) 파일을 다운로드합니다.  

<img src="./contents/korean-cloth.jpeg" width="300">

[claim-form.png](./contents/claim-form.png)를 업로드하면 아래와 같이 이미지에 대한 설명을 Text로 얻을 수 있습니다.

![image](https://github.com/kyopark2014/llm-multimodal-and-rag/assets/52392004/90c50e9e-5a13-4529-83bf-f2d1d1d544b7)

[flow_chart.png](./contents/flow_chart.png) 파일을 다운로드합니다.  

<img src="./contents/flow_chart.png" width="300">

[flow_chart.png](./contents/flow_chart.png)을 업로드하면 아래와 같이 flow chart에 대한 설명을 확인할 수 있습니다.

![image](https://github.com/kyopark2014/llm-multimodal-and-rag/assets/52392004/992e3ac7-cfa5-453f-94d8-135180fcac97)



### RAG 활용하기

"Conversation Type"에서 [RAG - opensearch]을 선택한 후에, [error_code.pdf](./contents/error_code.pdf)을 다운로드하여 채팅창 아래의 파일 아이콘을 이용하여 업로드합니다. 이후 아래처럼 채팅창에 "보일러 에러코드에 대해 설명하여 주세요."을 입력하면, 아래와 같은 결과를 얻을 수 있습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/d962beb6-f72f-4732-a9a6-583d51fd6c3c)


### 코드 요약하기

[lambda_function.py](./lambda-chat-ws/lambda_function.py)을 다운로드 후에 채팅창 아래의 파일 아이콘을 선택하여 업로드합니다. lambda_function.py가 가지고 있는 함수들에 대한 요약을 보여줍니다.

이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/eee5b660-cfcd-46ba-a21d-b8ce312efe3c)



### 문장 오류 확인

"Conversation Type"으로 [Grammer Error Correction]을 선택하고, "다음의 문문장에서 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."로 입력했을때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/63294a0f-f806-43be-bfed-ad9b140a0dde)

"In the following sentence, find the error in the sentence and aa explain it, and add the corrected sentence at the end of your answer."로 입력했을 때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/dfa3a8a0-2557-4a2f-9fca-6e531557725d)

### Agent 동작 확인

메뉴에서 Agent-ReAct를 선택합니다. 

"크리스마스까지 몇일 남은 거야?"라고 입력하면 아래와 같이 현재 시간을 기준으로 계산한 날짜를 알려줍니다. 

<img width="849" alt="image" src="https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/3a6bfa24-ab64-4f7d-acff-7403cab42883">

"서울과 부산의 온도를 비교해줘"라고 입력 합니다. 

<img width="848" alt="image" src="https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/2be560c4-51c6-4b39-ab21-5cdac15cb680">

LangSmith의 로그를 확인하면 아래와 같이 "Seoul", "Busan"의 온도를 get_weatehr_info를 이용해 확인한 결과를 도출합니다.

![image](https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/6e222410-6819-4050-a210-e3cc82397058)

"서울 여행에 대한 책을 추천해줘"로 입력후 결과를 확인합니다.

<img width="848" alt="image" src="https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/0213de6b-2580-4598-a2fc-b671aea43a37">

아래와 같이 get_book_list를 이용해 얻어온 도서 정보와 search_by_tavily로 얻어진 정보를 통합하였음을 알 수 있습니다.

![image](https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/6b33eb2d-11bc-4959-81d0-9ba76ca55ab2)

"보일러 에러코드에 대해 설명해줘."라고 입력하몬 RAG에서 얻어진 결과를 이용해 아래와 같이 답변합니다.

<img width="852" alt="image" src="https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/16ee0cdc-73d2-4e03-9d23-129b209af4ea">



## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "api-chatbot-for-llm-multimodal-on-aws", "api-llm-multimodal-on-aws"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.


```text
cd ~/environment/multimodal-on-aws/cdk-multimodal/ && cdk destroy --all
```


## 결론

LangChain을 이용하여 Anthropic Claude3.0으로 Multimodal과 RAG를 구현하였습니다. LangChain의 ChatBedrock을 활용하였고, Chain을 이용하여 Prompt를 구성하는 방법에 대해 설명하였습니다. 또한 OpenSearch를 이용하여 RAG를 구성하고 대규모로 문서를 처리하기 위한 event driven architecture에 대해 설명하였습니다. 


## 참고사항

### Debug 모드 종료하기

메시지 또는 파일 선택시에 아래와 같이 통계 정보를 제공하고 있습니다. 통계를 위해 추가적인 동작을 하므로 빠른 동작을 위해 Debug Mode를 disable할 필요가 있습니다. [cdk-multimodal-stack.ts](./cdk-multimodal/lib/cdk-multimodal-stack.ts)에서 "debugMessageMode"를 "false"로 설정하거나 채팅 창에서 "disableDebug"라고 입력합니다.

![image](https://github.com/kyopark2014/llm-multimodal-and-rag/assets/52392004/f73cfbe8-68ac-4bd5-b9e1-9af250684b75)
