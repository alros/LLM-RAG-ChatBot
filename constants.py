models = {
    'mistral': {
        'model_name': 'mistral',
        # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
        'prompt': """<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s> 
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]"""
    },
    'llama2': {
        'model_name': 'llama2',
        # https://smith.langchain.com/hub/rlm/rag-prompt-llama
        'prompt': """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
Question: {question} 
Context: {context} 
Answer: [/INST]"""
    }
}

model = 'mistral'