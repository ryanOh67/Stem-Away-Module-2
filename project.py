#from bs4 import BeautifulSoup
#import urllib.request as req

#res = req.urlopen('https://cafe.naver.com/mjbox/689285')
#soup = BeautifulSoup(res, 'html.parser')
#print(soup)
#print(soup.p)
#import csv
#from urllib.request import urlopen
#from bs4 import BeautifulSoup

#url = 'https://community.mycroft.ai/t/welcome-to-the-mycroft-community/8'
#html = urlopen(url).read()
#soup = BeautifulSoup(html, 'html.parser')

#otal = soup.find_all(soup.find_all(attrs={"class": "title raw-link raw-topic-link"}))


#for i in date:
#    print(i.attrs["datetime"]) #algorithm for finding the dates of the posts

from bs4 import BeautifulSoup
import csv
# i WANT to find correlation between the length of the post and the like    

from bs4 import BeautifulSoup
import urllib.request as req

url = 'https://community.mycroft.ai/t/welcome-to-the-mycroft-community/8'
res = req.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')




comments = soup.findAll('div',{'class': 'post', 'itemprop': 'articleBody', 'p': ''})

#comments = soup.findAll('div',{'class': 'post', 'itemprop': 'articleBody'})
comments.pop(0)
#for i in comments:
   # print((i.get_text()))           #algorithm for finding all the texts

commentLikes = soup.findAll(class_= "post-likes")
commentLikes.pop(0)
#for i in commentLikes:
    #if i.get_text() == "":
        #print(0)
   # else:
        #print(i.get_text())
    

x =1

file = open('helloRyan.csv','w', -1, "utf-8", newline = '')
writer = csv.writer(file)

writer.writerow(['Length', 'Likes', 'Date', 'Post'])

date = soup.findAll(class_= "post-time")
date.pop(0)




counter = 0

for i in comments:
    sampleComment = i.get_text()
    commentlength = len(sampleComment)



    if commentLikes[counter].get_text() == "":
        writer.writerow([commentlength, "0 Like", date[counter].get_text(),sampleComment])
    else:
        commentpostlike = commentLikes[counter].get_text()
        writer.writerow([commentlength, commentpostlike, date[counter].get_text(), sampleComment])

    counter= counter+1



  




file.close()




