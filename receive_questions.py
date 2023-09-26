# backend 에서 돌아갈 파이썬 스크립트

from ask_questions_rerank import AskQuestionsRerank


class RecvQuestions:
    def set_keys(self, openai_key, pinecone_key):
        self.ask_q = AskQuestionsRerank(openai_key, pinecone_key)

        return True


    #def recv_question(self, question, isFirst):
    def recv_question(self, question):
        result = self.ask_q.ask_first(question)
                        
        return result