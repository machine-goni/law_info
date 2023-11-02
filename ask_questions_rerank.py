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
import time



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
        # 위의 이유로 gpt-3.5-turbo 로 초반 사용했으나 completion 의 길이가 문제가 되어 답변이 잘려 파싱에러로 이어지는 경우가 꽤 생긴다.
        # 감안해서 본문을 더 많이 자르면 되지만 퀄리티의 문제를 고려 안할 수 없기 때문에 일단 16k 를 쓰고 상황을 봐서 4k 로 돌아갈지 결정한다.
        self.model_type = 1     # 0: "gpt-3.5-turbo", 1: "gpt-3.5-turbo-16k"
        if self.model_type == 1:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.0, openai_api_key=self.openai_key)
        else :
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=self.openai_key)
    

    def ask_first(self, query):

        # exception 이 나서 보여줄께 없을땐 그냥 retriever 가 찾아낸 결과라도 보여주기 위해
        retriever_case_no_list = []
        retriever_url_list = []    
            
        try:      
            # 원래 3개를 찾았는데 ChatGPT 속도가 느려져서 rerank 로 3개를 찾으면 cloudtype 에서 timeout 이 걸려버린다.       
            nearest_k = 2   #3
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k})                         
            
                     
            # llm 으로 보내서 rerank 를 하기위해 retriever 로 문서를 추려낸다. 
            doc_max_len = 3970
            prompt_len = 330
            completion_len = 350

            if self.model_type == 1:    # 3.5 turbo 16k model
                doc_length_limit = 12000
            else :                      # 3.5 turbo 4k model
                doc_length_limit = doc_max_len - prompt_len - completion_len    # 이 이상의 길이는 참조 문서에서 앞에서부터 자른다. 안그럼 토큰오바로 에러난다.
            
            # retriever 검색
            relevant_docs = self.retriever.get_relevant_documents(query)

            for relevant_doc in relevant_docs:
                # 자르기 전
                #print(f"No: {relevant_doc.metadata['source']}, Name: {relevant_doc.metadata['case_no']}")                  
                #print(relevant_doc.page_content)

                retriever_case_no_list.append(relevant_doc.metadata['case_no'])
                retriever_url_list.append(f"https://www.law.go.kr/DRF/lawService.do?OC=xivaroma&target=prec&ID={relevant_doc.metadata['source']}&type=html")

                relevant_doc.page_content = relevant_doc.page_content.replace("<br/>", "")
                if len(relevant_doc.page_content) > doc_length_limit:                    
                    need_cut_len = len(relevant_doc.page_content) - doc_length_limit                    
                    new_page_content = relevant_doc.page_content[need_cut_len:]
                    #print(f"new_page_content length: {len(new_page_content)}")
                    #print(f"new_page_content : {new_page_content}")
                    relevant_doc.page_content = new_page_content
            
            # create chain
            map_rerank_chain = load_qa_chain(
                llm = self.llm,
                chain_type='map_rerank',                
                return_intermediate_steps = True, # 이걸 넣지 않으면 score 를 얻을 수가 없다. ConversationalRetrievalChain 에는 이걸 넣을 수가 없다.
                verbose = True
            )

            # 추려낸 문서를 input_documents 에 넣어준다. llm 의 호출수는 input_documents 수 만큼 늘어난다. 
            # 종종 영어로 답변하는 경우가 있어서 넣어준다. custom prompt 는 rerank 에선 쓸 수 없기때문에 question 에 붙여준다.
            # 주의 할 점은 질문의 맨 앞에 붙여줘야 유효하고 retriever 가 검색을 끝마친 후 llm 에게 보내기 직전에 붙여준다.         
            to_korean = f'한국어로 답해라. '
            response = map_rerank_chain({'input_documents':relevant_docs, 'question':f'{to_korean}{query}'})
            
            # score 로 반환할 답변을 거른다
            cut_line = 0   #70  #무조건 답변이 나오도록 점수를 낮춘다. 대신 낮은 점수는 낮은 관련도라고 알려주도록 한다.
            rslt_case_no_list = []
            rslt_url_list = []
            rslt_score_list = []
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
                            rslt_score_list.append(score)

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
            json_data["scores"] = rslt_score_list
            json_data["answers"] = rslt_answer_list
            json_data["final_answer"] = rslt_final_answer   

            """
            # test
            for i in range(0, len(rslt_case_no_list)):
                print(f'case {i}:')
                print(f'case no: {rslt_case_no_list[i]}')
                print(f'url: {rslt_url_list[i]}')
                print(f'score: {rslt_score_list[i]}')
                print(f'answers: {rslt_answer_list[i]}')
                print('\n')

            print(f'final answer: {rslt_final_answer}')
            """

            #now = time
            #print(f"현재: {now.localtime().tm_mon}월{now.localtime().tm_mday}일 {now.localtime().tm_hour}:{now.localtime().tm_min}")
            #print(response)
            return json.dumps(json_data)
                                    

        except Exception as e:
            print("Exception!!:" + str(e)) 
            
            json_data = {}
            json_data["Exception"] = str(e)
            json_data["cases"] = retriever_case_no_list
            json_data["urls"] = retriever_url_list

            #print(json_data)

            # 정상 응답과 형식을 맞추기 위해 dictionary 형태로 보낸다.
            # json 형식으로 보내기 위해 json.dumps 를 사용해 dictionary 를 json 으로 변환
            return json.dumps(json_data)                                
    


def decrypt(key, ciphertext):
    decoded_ciphertext = base64.b64decode(ciphertext)
    iv = decoded_ciphertext[:AES.block_size]
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(decoded_ciphertext[AES.block_size:]), AES.block_size).decode('utf-8')
    
    return plaintext
