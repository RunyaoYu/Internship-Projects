"""
Author: Runyao Yu
runyao.yu@tum.de
Research Internship in ETH Zurich
For Academic Use Purpose only
"""

import pandas as pd
import requests
import time
import re
from urllib.request import urlretrieve
import os
import json

zid = []  # 保存用户id save user id
ztime = []  # 保存用户发表评论的时间 save the time when users post
zname = []  # 保存用户姓名 save user's name
zanswer = []  # 保存用户评论内容 save posts
zcomment = []  # 用户评论数 save the number of comments
zsupport = []  # 用户点赞数 save the like number of posts
totals = []  # 记录评论的总条数 save the total number of comments
zfollower = []
zans_id = []
zself_pre = []  # 保存答主的po文 save the main post

# path = 'C:\\Users\\Administrator\\Desktop\\ETH\\pachong_test' # if in mac system, change \\ to /
path = "/Users/rampageyao/Desktop/eth/eth project"

def zhuhuSipder(page):
    url = "https://www.zhihu.com/api/v4/questions/264508629/answers"
    # url = "https://www.zhihu.com/api/v4/questions/59942156/answers"

    # 必须添加cookie 信息
    # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:57.0) Gecko/20100101 Firefox/57.0',
    #            "cookie": '_zap=3c8dd525-79f4-49a2-8868-997f15e3940a; d_c0="AFCtonbmLhCPTpJfXUoXdgNQtsDEv-HF6EM=|1570805329"; _xsrf=3Nr0gMj5zL7Z4p61y7xdsWEU2E3DeDDe; tst=r; _ga=GA1.2.1669020552.1582625102; __utmv=51854390.100-1|2=registration_date=20120207=1^3=entry_date=20120207=1; q_c1=5e2bd78c7427447b9471f6d3582c3a32|1609171411000|1573463421000; __utmz=51854390.1611145402.4.3.utmcsr=zhihu.com|utmccn=(referral)|utmcmd=referral|utmcct=/question/338460859; __utmc=51854390; __utma=51854390.1669020552.1582625102.1608895944.1611145402.4; capsion_ticket="2|1:0|10:1611653148|14:capsion_ticket|44:NjJhYTEwOGM5MTQ3NGFlY2I5YzliMDNmMjg3ZWVkNjM=|8141c78ce0df68710c4c0098498c55b0201349772c1a60c8130b474fc42a37c8"; z_c0="2|1:0|10:1611653242|4:z_c0|92:Mi4xeGtvREFBQUFBQUFBVUsyaWR1WXVFQ2NBQUFDRUFsVk5lbTAzWUFDLUxhTkUwcW0xSWlYbHhvQTJHM0dEblpTbmJB|dec99b05dd9fdd3412b6f06a4c7a0790b1a8b2572ad174ec10dc08db6452c7a4"; SESSIONID=1pBufNyUNhQSChZFZKEEDgJEeM2moy968PVKM9NoFXG; JOID=UFkQBU9LTaXItOPZF0-Y9dz1v1IBeTnkv9SupEEDJvCY9Y2AVYZAtKyx5N4WgANso8gnZhO-7bnjTy0uTXQZe7U=; osd=V18cBkhMS6nLs-TfG0yf8tr5vFUGfzXnuNOoqEIEIfaU9oqHU4pDs6u36N0RhwVgoM8gYB-96r7lQy4pSnIVeLI=; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1611666062,1611666198,1611666199,1611667243; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1611668755; KLBRSID=fe78dd346df712f9c4f126150949b853|1611668756|1611664070',
    #            }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
               "cookie": '_xsrf: 0UITMf3bZBvoIoEEf4BA0IqvdZuj3d7m; KLBRSID: 57358d62405ef24305120316801fd92a|1628451797|1628451797',
               }

    # Cookie：用于维持用户当前的会话，通常会有多个cookie
    # Referer：表示请求从那个页面发过来的
    # User-Agent：它为服务器提供当前请求来自的客户端的信息，包括系统类型、浏览器信息等。这个字段内容在爬虫程序种非常重要，
    # 我们通常可以使用它来伪装我们的爬虫程序，如果不携带这个字段，服务器可能直接拒绝我们的请求。

    data = {  # 这个data  就是 xhr 中 的查询参数 this data is the query existed in xhr
        "include": "data[*].is_normal,admin_closed_comment,reward_info,is_collapsed,annotation_action,annotation_detail,collapse_reason,is_sticky,collapsed_by,suggest_edit,comment_count,can_comment,content,editable_content,voteup_count,reshipment_settings,comment_permission,created_time,updated_time,review_info,relevant_info,question,excerpt,relationship.is_authorized,is_author,voting,is_thanked,is_nothelp,is_labeled,is_recognized,paid_info,paid_info_content;data[*].mark_infos[*].url;data[*].author.follower_count,badge[*].topics",
        "limit": "5",  #
        "offset": str(page),
        "platform": "desktop",
        "sort_by": "default"
    }
    data_html = requests.get(url=url, params=data, headers=headers).json()  # 返回 json 信息， 5个用户一页 back json info, 5 users pro page

    # 将毫秒数变成日期格式 change units in ms to units in date
    def timestamp_to_date(time_stamp, format_string="%Y-%m-%d %H:%M:%S"):
        time_array = time.localtime(abs(time_stamp))
        str_date = time.strftime(format_string, time_array)
        return str_date
    # print(timestamp_to_date(data_html[0]["created_time"]))

    n = 0
    for i in data_html["data"]:
        n = n + 1
        id = i['author']['id']
        name = i['author']['name']
        answer = i["content"]  # po主得回答，包含照片和文本信息，不是网友得评论 this section only contains main post and images excluding sub posts 

        # self_pre = i["excerpt"]  # 这一段是self presentation, 但只是包含部分内容，而不是全部内容 this section only contains self presentation
        # 截取po主的回答 get reply of the comment
        REG = re.compile('<[^>]*>')
        self_pre = REG.sub("", answer).replace("\n", "").replace(" ", "")

        comment = i["comment_count"]
        support = i["voteup_count"]
        follower = i['author']['follower_count']
        print(i["created_time"])
        time_ = timestamp_to_date(i["created_time"])
        answer_id = i['id']
        current_path = path + "/" + str(page+n-1) + "_" + str(name) + "_" +id # 文件夹的名字，目前是以po主的id当作尾缀 name of file folder, ended with user ID name
        os.makedirs(current_path)

        file_txt = open(current_path + "/" + "self presentation" + ".txt", 'w', encoding='utf-8')
        file_txt.write(self_pre)
        file_txt.close()

        reg = r'data-actualsrc="(.*?)"'
        imgRe = re.compile(reg, re.S)
        imgUrls = imgRe.findall(str(answer))
        number = 0
        for imgUrl in imgUrls:
            try:
                splitPath = imgUrl.split('.')
                fTail = splitPath.pop()
                if len(fTail) > 3:
                    fTail = 'jpg'
                fileName = current_path + "/" + str(number) + "." + fTail
                urlretrieve(imgUrl.split('?', 1)[0], fileName)
                number += 1
            except:
                number += 1
        j = 0
        comment_id = []
        comment_name = []
        comment_content = []
        comment_time = []

        while True:
            comment_url = 'https://www.zhihu.com/api/v4/answers/' + str(answer_id) + '/root_comments?order=normal&limit=20&offset={}&status=open'.format(j)
            j += 20
            try:
                res = requests.get(comment_url, headers=headers).content.decode('utf-8')
                jsonfile = json.loads(res)
                next_page = jsonfile['paging']['is_end']
            except:
                break
            for data in jsonfile['data']:
                c_id = data['author']['member']['id']
                c_name = data['author']['member']['name']
                c_content = data['content']
                c_time = timestamp_to_date(data['created_time'])
                comment_id.append(c_id)
                comment_name.append(c_name)
                comment_content.append(c_content)
                comment_time.append(c_time)
            if next_page == True:
                break

        v = list(zip(comment_id, comment_name, comment_time, comment_content))
        pd.DataFrame(v, columns=["id", "name", "time", "content"]).to_excel(current_path+"/comments.xlsx")
        zid.append(id)
        ztime.append(time_)
        zname.append(name)
        zanswer.append(answer)
        zcomment.append(comment)
        zsupport.append(support)
        zfollower.append(follower)
        zans_id.append(answer_id)

    totals_ = data_html["paging"]["totals"]  # 评论总条数 total number of comments
    totals.append(totals_)
    # print(totals[0])
    return totals[0]


# 多页爬虫 multipage
def mulitypage():
    page = 0
    zhuhuSipder(page)
    time.sleep(10)
    while (page < totals[0]):
        print("正在抓取第{}页".format(int(page / 5)))
        page += 5
        zhuhuSipder(page)


# 保存数据 save data
def savedata():
    v = list(zip(zid, ztime, zname, zanswer, zans_id, zcomment, zsupport, zfollower))
    print(v)
    pd.DataFrame(v, columns=["id", "time", "name", "answer", "answer_id", "comment", "support", "follower"]).to_excel("zhihu_comment_9000.xlsx")


if __name__ == "__main__":
    mulitypage()
    savedata()