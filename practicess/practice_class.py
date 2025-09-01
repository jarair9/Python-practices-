# making image generation using class
import requests


class news:
    def __init__(self,url,param):
        self.url = url
        self.param = param

    def location(self,loc):
        self.loc = loc
        for words in self.param.items():
            if words[1] == []:
                words[1].append(self.loc)


    def res(self):
        response1 = (requests.get(self.url, self.param))
        news_data = response1.json().get("articles", [])[:1]
        for articles in news_data:
            content = articles.get("content")
            title = articles.get("title")
            description = articles.get("description")
            url = articles.get("url")
            print(f"Title: {title}")
            print(f"Content: {content}")
            print(f"Description: {description}")
            print(f"URL: {url}")


url = "https://newsapi.org/v2/everything"
param = {
    "q": [],                                      
    "sortBy": "publishedAt",
    "pageSize": 1,                          
    "language" : "en",         
    "apiKey": "cae1da4e00394a9c9406cb97b3a8049f"   
}
        
N = news(url,param)
loc = str(input("Enter the loction: "))
N.location(loc)
N.res()        
        

        