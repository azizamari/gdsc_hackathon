from langchain_alt import BSHTMLLoader, TextLoader
from langchain.chains.summarize import load_summarize_chain
import requests
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import json
load_dotenv()
import os

llm=ChatOpenAI(model="gpt-3.5-turbo-0613")
test_url = "https://boredgeeksociety.beehiiv.com/p/ai-weekly-digest-14-ai-lowcode-is-a-productivity-game-changer"
def get_html_content(url=test_url):

    headers = {
        'authority': 'boredgeeksociety.beehiiv.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-GB,en;q=0.9',
        'cache-control': 'max-age=0',
        'cookie': 'orchid_page_view:38ef7703-652e-469a-b80f-43ccbaeac0bf=IjE5NjA4OGM1LWJkN2UtNGJmNS05M2EzLWY0MDM5NTk5NGFjZSI%3D; visit_token=IjE5NjA4OGM1LWJkN2UtNGJmNS05M2EzLWY0MDM5NTk5NGFjZSI%3D; _orchid_session=eyJjc3JmIjoiNzM0NzVjNmYtM2I0NC00MzcwLWFmMzEtZDMxNmIxZDMxNDI2IiwicHVibGljYXRpb24iOnsiaWQiOiJmNTgwZjIxNC05MmFmLTRiNGMtYWU1NS0wMjI5YjBmNDBlNGUiLCJwcmVtaXVtRW5hYmxlZCI6ZmFsc2UsImhhc1JlZmVycmFsUHJvZ3JhbSI6ZmFsc2UsIm5hbWUiOiJCZXlvbmQgVGhlIEFJIEhvcml6b24iLCJsYW5ndWFnZSI6ImVuIn0sInRva2VuIjpudWxsfQ%3D%3D.0zpl1oQRPnAAExgxYqk7IbvjyWqhDmaUmDmJdAuFCFE; __cf_bm=o9kVBLzVw91CWZx7ZkNJBI2nPVfh4D1vlxtD4wItP8o-1686655010-0-AWFsmhN/DtQlMuEhyLS+UPMvyVOVgO5FsdQG7DJTE4SVcNustQsVFZ9ilXNhBsq8L/ptZ30AMnjUgmEBcaybTns=; _ga=GA1.2.922378055.1686655054; _gid=GA1.2.1909993690.1686655054; _gat_UA-199090589-3=1; ln_or=eyI0MjEyOTg1IjoiZCJ9; _fbp=fb.1.1686655056055.953409685',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        html_content = response.text
        return html_content
    else:
        return 'Error'
    
def get_text_content(content):
    bshtml_loader = BSHTMLLoader(html_text=content)
    document = bshtml_loader.load()
    text_content = document[0].page_content
    return text_content
def get_text_summary(text):
    loader=TextLoader(text=text)
    docs=loader.load()
    text_splitter = TokenTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    docs = text_splitter.split_documents(docs)
    # for doc in docs:
    #     template_clean="""
    #     Clean up this text, I want to hide the fact that this is a blog/ article. Make it raw text.
    #     Remove any phrases like "newsletter" or "Read more" or "Continue reading" or "Share this post" or "In this blog" or "In this article"
    #     Get the import information from this text and remove unnecessary information.
    #     Summarize while keeping 50% of the text
    #     Text: {text}
    #     Cleaned text:
    #     """      
    #     prompt_clean=PromptTemplate(
    #         input_variables=["text"],
    #         template=template_clean
    #     )
    #     chain_clean=LLMChain(
    #         llm=ChatOpenAI(model="gpt-3.5-turbo-0613"),
    #         prompt=prompt_clean,
    #     )  
    #     doc.page_content=chain_clean.run(doc.page_content)
    #     print("aaaaaaaaaaaaa")
    # print(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)
def get_main_points(input):
    prompt=PromptTemplate(input_variables=['text'],template="This is a blog's summary: {text}\nI need to extract the three main topics from this blog. Make sure each topic is unique.\n\nFormat: JSON OBJECT {{\"topics\":[topic1 sentence, topic2 sentence, topic3 sentence]}}")
    points_chain=LLMChain(llm=llm, prompt=prompt)
    res=points_chain.run(input)
    # try parsing json
    try:
        res=json.loads(res)
        return res.get('topics')
    except:
        print('Error parsing json')
        print(res)
        pass
def get_title(input):
    prompt=PromptTemplate(input_variables=['text'],template="This is a blog's summary: {text}\nI need to extract a title  that can i use in order to create slides about the topic  about  this blog.\n\nFormat: JSON OBJECT {{\"title\":[short sentence which is the title]}}")
    points_chain=LLMChain(llm=llm, prompt=prompt)
    res=points_chain.run(input)
    # try parsing json
    try:
        res=json.loads(res)
        return res.get('title')
    except:
        print('Error parsing json')
        print(res)
        pass
def generate_post(topic):
    db=pickle.load(open('vectorstore.pkl','rb'))
    docs = db.similarity_search(topic)
    prompt=PromptTemplate(
        input_variables=['context','topic'],
        template="""Write a brief instagram post about {topic}.\n\nContext: {context} \n\n and the post has to have to have a json format based on the structure below :
        Structure: {{"title":"","text":"", "tags":["tag1","tag2",...]}}"""
    ) #TODO: add tone,..
    chain=LLMChain(llm=llm,prompt=prompt)
    res=chain.predict(context=docs[0].page_content,topic=topic)
    return res

## script generation

template="""Create an engaging 40 to 60 seconds video going over this topic. 
The first sentence  need to instantly capture attention and curiosity. 
Be as concise in your language as possible due to the time constraints. 
Make every sentence compelling. 
Don't mention that this is a blog, article or similar. This is a youtube video.
Don't include call to actions like visiting website, subscribing, etc.
Write in raw text and if you want the voice over artist to take a pause, use this token <PAUSE>. 
Here is some information about the topic that may help in your script
Topic: {information}
Use this information to create a script that is engaging, interesting and allows user to learn new things about the topic.
Format: return a Json item {{"CatchyLine": sentence, "Script": 40 to 60 seconds script}}
"""

prompt=PromptTemplate(
    input_variables=["information"],
    template=template
)

chain=LLMChain(
    llm=ChatOpenAI(model="gpt-3.5-turbo-0613"),
    prompt=prompt,
)

def generate_script(text):
    return chain.predict(information=text)