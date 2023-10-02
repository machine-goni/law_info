# pip install tiktoken
# pinecone 을 pip install 시 pinecone-client 으로 인스톨하고 requirement 에도 pinecone-client 로 넣어줘라

'''
이 코드는 chain_type 을 기존의 "stuff" 가 아니라 "map_rerank" 를 사용하기 위한 코드이다.
rerank 는 retriever 가 검색한 문서를 llm 에 보내고, llm 이 query 에 대해 각각의 문서에서 답을 찾고
답변하기에 얼마나 적합한지에 대한 점수를 산출하여 반환하고 최종적으로 가장 높은 점수를 받은 문서에 대한 답을
보내준다. 하지만 어떤때는 최고 점수의 문서에 대한 답만을 보내는 것이 아니라 높은 점수의 답들을 요약해서 보내줄때도 있다.
verbose=True 를 해보면 llm 에 보내는 prompt 에 few shot 을 덧붙여 보내는 것을 볼 수 있는데,
이런 부가적인 prompt 추가가 보내는 수의 문서만큼 매번 발생하고, 마찬가지로 보내는 수의 문서만큼 llm 으로의 prompt + completion 이 
발생하기 때문에 그 만큼 호출수와 token 이 더 소모된다.
llm 이 문서를 보고 직접판단하기 때문에 좀 더 높은 정확도를 보이긴 하지만 llm 으로 보내지는 문서의 수가 많이 않을 때는
그다지 큰 차이를 못 볼 수 있다.
가장 큰 문제는 llm 이 종종(어쩔때는 빈번하게) langchain 이 보내는 few shot 을 따르지 않을 때가 있다.
그렇게 되면 langchain 이 llm 으로 부터 기대하는 형태를 받지 못해 parsing error 를 뱉어 버린다.

이 코드에서는 ConversationalRetrievalChain 을 사용하지 않고 load_qa_chain 을 썻는데,
ConversationalRetrievalChain 에서 chain_type = "map_rerank" 로 사용할 순 있지만 이렇게 되면 
각 문서에 대한 scrore 를 얻을 수가 없다(return_intermediate_steps=True 이렇게 해야 얻을 수 있지만 
해당 속성이 없고 combine_docs_chain_kwargs 에 넣을 수 있을것 같지만 넣으면 에러난다). 
그래서 load_qa_chain 을 쓸 수 밖에 없다. 확인은 해보지 않았지만 map_rerank 에서는 prompt template 을
사용할 수 없을 것 같다. langchain 이 few shot 을 통해 요구하는 포맷이 있기때문에 충돌이 예상된다.
'''


import os
import json
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64



PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west4-gcp-free')
INDEX_NAME = "law-info-precedents-under-3970"
sk = 'rkskekfkakqktkdmgpdmgpdmgpgw2950'



class AskQuestionsRerank:
    
    def __init__(self, openai_key, pinecone_key):
        self.openai_key = decrypt(sk, openai_key)
        self.pinecone_key = decrypt(sk, pinecone_key)        

        # create embedding instance
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)

        # initialize pinecone
        pinecone.init(api_key=self.pinecone_key, environment=PINECONE_API_ENV)

        # 이미 문서를 임베딩해서 pinecone vector store 에 넣었다면 거기에서 끌어다 쓰면 된다
        self.vectorstore = Pinecone.from_existing_index(INDEX_NAME, self.embeddings)
    
        # 판례의 특성상 내용이 많아 잘게 쪼갤경우 판례의 내용과 결론이 쪼개겨 엉뚱한 답을 낼 가능성이 있어 chunk size 를 4000 으로 하였다.
        # 따라서 token 제한을 피하기 위해 gpt-3.5-turbo-16k 를 쓰려했으나 rerank 는 문서마다 따로 호출하여 답변을 얻는 것이기 때문에
        # 굳이 2배 비싼 16k 를 쓸 필요는 없어 보인다. 그래서 그냥 gpt-3.5-turbo 를 쓴다.
        # 위와 같은 이유로 gpt-3.5-turbo 쓰려고 했으나 rerank 특성상 붙는 few shot 등 때문에 4096 토큰을 넘는 경우가 생겨 gpt-3.5-turbo-16k 를 사용.
        #self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=self.openai_key)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.0, openai_api_key=self.openai_key)
    

    def ask_first(self, query):
            
        try:             
            nearest_k = 3            
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k})                         
            
            # llm 으로 보내서 rerank 를 하기위해 retriever 로 문서를 추려낸다. 
            relevant_docs = self.retriever.get_relevant_documents(query)            
            #for relevant_doc in relevant_docs:
            #    print(f"No: {relevant_doc.metadata['source']}, Name: {relevant_doc.metadata['case_no']}")  
            
            # create chain
            map_rerank_chain = load_qa_chain(
                llm = self.llm,
                chain_type='map_rerank',                
                return_intermediate_steps = True, # 이걸 넣지 않으면 score 를 얻을 수가 없다. ConversationalRetrievalChain 에는 이걸 넣을 수가 없다.
                verbose = True
            )

            # 추려낸 문서를 input_documents 에 넣어준다. llm 의 호출수는 input_documents 수 만큼 늘어난다. 
            response = map_rerank_chain({'input_documents':relevant_docs, 'question':query})            
            
            # score 로 반환할 답변을 거른다
            cut_line = 70
            rslt_case_no_list = []
            rslt_url_list = []
            rslt_answer_list = []
            rslt_final_answer = ''

            if 'intermediate_steps' in response:
                intermediate = response['intermediate_steps']
                
                for i, res in enumerate(intermediate):
                    if 'score' in res:
                        score = res['score']

                        if int(score) >= cut_line:
                            rslt_case_no_list.append(relevant_docs[i].metadata['case_no'])
                            rslt_url_list.append(f"https://www.law.go.kr/DRF/lawService.do?OC=xivaroma&target=prec&ID={relevant_docs[i].metadata['source']}&type=html")

                            if 'answer' in res:
                                answer = res['answer']
                                # 안드로이드에서 콤마 기준으로 분할할거라서 바꿔줘야 한다.
                                rslt_answer_list.append(answer.replace(',', '||'))
                                #print(f'intermediate answer {i}: {answer}')                                

                            #print(f'intermediate score {i}: {score}')

            if 'output_text' in response:                      
                rslt_final_answer = response['output_text']
                #print(f'Final Answer: {rslt_final_answer}')

            json_data = {}
            json_data["cases"] = rslt_case_no_list
            json_data["urls"] = rslt_url_list
            json_data["answers"] = rslt_answer_list
            json_data["final_answer"] = rslt_final_answer   

            """
            # test
            for i in range(0, len(rslt_case_no_list)):
                print(f'case {i}:')
                print(f'case no: {rslt_case_no_list[i]}')
                print(f'url: {rslt_url_list[i]}')
                print(f'answers: {rslt_answer_list[i]}')
                print('\n')

            print(f'final answer: {rslt_final_answer}')
            """

            return json.dumps(json_data)
            #print(response)
            #return response                                    
            

        except Exception as e:
            print("Exception!!:" + str(e))            
            
            json_data = {}
            json_data["final_answer"] = "Exception!!:" + str(e)

            # json 형식으로 보내기 위해 json.dumps 를 사용해 dictionary 를 json 으로 변환
            #return json.dumps(json_data)
            
            # 정상 응답과 형식을 맞추기 위해 dictionary 형태로 보낸다.
            return json_data
    


def decrypt(key, ciphertext):
    decoded_ciphertext = base64.b64decode(ciphertext)
    iv = decoded_ciphertext[:AES.block_size]
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(decoded_ciphertext[AES.block_size:]), AES.block_size).decode('utf-8')
    
    return plaintext
