#!/bin/sh

alarm_percent=80
while true
do
    root_dir="/dev/vda1"
    data_dir="/dev/vdb"
    root_dir_percent=`df -h | grep "${root_dir}" | awk '{print $5}' | sed 's/%//g'`
    data_dir_percent=`df -h | grep "${data_dir}" | awk '{print $5}' | sed 's/%//g'`
    has_byte_dance_report=`ps -ef | grep byte_dance_report | grep -v grep  | wc -l `
    has_ucdn_schedule_go=`ps -ef | grep ucdn-schedule-go | grep -v grep  | wc -l `
    has_ucdn_api_go=`ps -ef | grep ucdn_api_go | grep -v grep  | wc -l `
    has_ucdn_api=`ps -ef | grep /root/ucdn-api/app.js | grep -v grep  | wc -l `
    if [ ${has_ucdn_api_go} -gt 0 ];then
        echo "ucdn_api_go strong"
    else
        curl --connect-timeout 5 -m 5 -X "POST" "http://cdn.admin.ucloud.com.cn:88/monitor.cgi" -H "Accept:application/json" -d "id=10329&title=ucdn进程告警&content=10.182.44.242 ucdn_api_go进程掉了 "
    fi
    if [ ${has_ucdn_api} -gt 0 ];then
        echo "ucdn-api strong"
    else
        curl --connect-timeout 5 -m 5 -X "POST" "http://cdn.admin.ucloud.com.cn:88/monitor.cgi" -H "Accept:application/json" -d "id=10329&title=ucdn进程告警&content=10.182.44.242 ucdn-api进程掉了 "
    fi
    if [ ${has_byte_dance_report} -gt 0 ];then
        echo "byte_dance_report strong"
    else
        curl --connect-timeout 5 -m 5 -X "POST" "http://cdn.admin.ucloud.com.cn:88/monitor.cgi" -H "Accept:application/json" -d "id=10329&title=ucdn进程告警&content=上报字节数据：byte_dance_report进程掉了 "
    fi
    if [ ${has_ucdn_schedule_go} -gt 0 ];then
        echo "ucdn-schedule-go strong"
    else
        curl --connect-timeout 5 -m 5 -X "POST" "http://cdn.admin.ucloud.com.cn:88/monitor.cgi" -H "Accept:application/json" -d "id=10329&title=ucdn进程告警&content=定时任务：ucdn-schedule-go进程掉了 "
    fi
    if [ $root_dir_percent -gt $alarm_percent  ];then
        curl --connect-timeout 5 -m 5 -X "POST" "http://cdn.admin.ucloud.com.cn:88/monitor.cgi" -H "Accept:application/json" -d "id=10330&title=ucdn机器磁盘告警&content=10.182.44.242 root磁盘使用占比超80% "
    fi
    if [ $data_dir_percent -gt $alarm_percent  ];then
        curl --connect-timeout 5 -m 5 -X "POST" "http://cdn.admin.ucloud.com.cn:88/monitor.cgi" -H "Accept:application/json" -d "id=10330&title=ucdn机器磁盘告警&content=10.182.44.242 data磁盘使用占比超80% "
    fi
    sleep 5m
done