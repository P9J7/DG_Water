# -*- coding:UTF-8 -*-
import csv
import requests
import json
import time
import datetime

if __name__ == '__main__':
    # file = open('test.json', 'w', encoding='utf-8')
    deviceid_list = [540085038, 927141434, 643567145, 793381679, 559876913, 559614777, 928321083, 927075899, 540019503,
                     910167583, 910167582, 912068127, 912068126, 793381684, 910167581, 928910907, 912068124, 928124484,
                     928779835, 912002591, 911347229, 927272506]
    start_time = "2019-12-20%2000:00:00"
    end_time = "2020-01-20%2000:00:00"
    i = 1
    for deviceid in deviceid_list:
        target = 'http://www.ecomonitor.com.cn/water/opendata/data?key=hj8d9bd2bcdsddgdsa1a166076xb9zX&startTime=' + \
                 start_time + '&endTime=' + end_time + '&deviceid=' + str(deviceid)
        print(target)
        req = requests.get(url=target)
        data = req.json()["result"]
        # json.dump(data, file, ensure_ascii=False, indent=2)
        # file.close()

        flag = True
        for item in data:
            if flag:
                # 获取地点
                places = list(data.keys())
                csvfile = open(places[0] + '.csv', 'w', newline='', encoding='utf-8')
                writer = csv.writer(csvfile)
                # 将属性列表写入csv中
                writer.writerow(
                    ["devicename", "deviceid", "latitude", "longitude", "time", "水质等级", "pH", "DO", "COD", "氨氮", "ORP",
                     "浊度", "叶绿素", "电导率", "水温"])
                for place in places:
                    # 某地数据, place_item
                    p_i = data[place]
                    # 直接读取部分数据
                    list_fix = [p_i["devicename"], p_i["deviceid"], p_i["latitude"], p_i["longitude"]]
                    # 某地数据datalist,place_item_list
                    p_i_l = p_i["dataList"]
                    for data_item in p_i_l:
                        data_list = list_fix.copy()
                        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data_item["time"] / 1000))
                        data_list.append(otherStyleTime)
                        data_item = data_item["items"]
                        # print(data_item)
                        dict_tmp = dict()
                        for item1 in data_item:
                            # data_list.append(item1["value"])
                            dict_tmp[item1["name"]] = item1["value"]
                        if "水质等级" or "WQI" in dict_tmp:
                            if "水质等级" in dict_tmp:
                                data_list.append(dict_tmp["水质等级"])
                            else:
                                data_list.append(dict_tmp["WQI"])
                        else:
                            data_list.append("NaN")
                        if "pH" in dict_tmp:
                            data_list.append(dict_tmp["pH"])
                        else:
                            data_list.append("NaN")
                        if "DO" in dict_tmp:
                            data_list.append(dict_tmp["DO"])
                        else:
                            data_list.append("NaN")
                        if "COD" in dict_tmp:
                            data_list.append(dict_tmp["COD"])
                        else:
                            data_list.append("NaN")
                        if "氨氮" in dict_tmp:
                            data_list.append(dict_tmp["氨氮"])
                        else:
                            data_list.append("NaN")
                        if "ORP" in dict_tmp:
                            data_list.append(dict_tmp["ORP"])
                        else:
                            data_list.append("NaN")
                        if "浊度" in dict_tmp:
                            data_list.append(dict_tmp["浊度"])
                        else:
                            data_list.append("NaN")
                        if "叶绿素" in dict_tmp:
                            data_list.append(dict_tmp["叶绿素"])
                        else:
                            data_list.append("NaN")
                        if "电导率" in dict_tmp:
                            data_list.append(dict_tmp["电导率"])
                        else:
                            data_list.append("NaN")
                        if "水温" in dict_tmp:
                            data_list.append(dict_tmp["水温"])
                        else:
                            data_list.append("NaN")
                        writer.writerow(data_list)  # 将属性列表写入csv中
                flag = False
        print("爬取完成" + str(i) + "个！")
        i += 1
        csvfile.close()
