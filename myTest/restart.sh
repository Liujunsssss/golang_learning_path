ps -ef | grep /data/monitor/moniter_alarm.sh | grep -v "grep" | awk '{print $2}' | xargs kill -9
nohup /data/monitor/moniter_alarm.sh &
ps -ef | grep /data/monitor/moniter_alarm.sh | grep -v "grep"