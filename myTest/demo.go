package main

import (
	"fmt"
	"log"
	"script/proto/umonitor2"
	"time"
	uflog "uframework/log"
	ufmessage "uframework/message"
	"uframework/message/protobuf/proto"
	ufnet "uframework/net"

	"github.com/jasonlvhit/gocron"
)

type ReqParamForm struct {
	UUID   string `json:"uuid"`
	ItemId int    `json:"item_id"`
	Value  int64  `json:"value"`
	Time   int64  `json:"time"`
}

func setLog() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	uflog.InitLogger(".", "SimulateThirdpart", ".log", 50, "INFO")
}

func main() {
	setLog()

	DoTask()

	sch := gocron.NewScheduler()
	sch.Every(1).Minute().Do(DoTask)
	<-sch.Start()
}

func DoTask() {
	req := &ReqParamForm{}
	uuids1 := make([]string, 0)
	for i := 0; i < 100; i++ {
		uuids1 = append(uuids1, fmt.Sprintf("ucdn-test-%05d", i+1))
	}

	itemIds1 := []int{
		44002,
	}

	value := int64(61)
	valueTime := time.Now().Unix()

	// set3
	ThirdpardRequest("10.189.151.143", 6507, uuids1, itemIds1, value, valueTime)
	// ThirdpardRequest("172.28.168.88", 6508, uuids2, itemIds1, value, valueTime)
	// ThirdpardRequest("172.18.176.136", 4444, uuids3, itemIds1, value, valueTime)

	// // to region fixed node
	// ThirdpardRequest("172.28.246.102", 6508, uuids3, itemIds1, value, valueTime)

	// // to regino 105
	// ThirdpardRequest("172.28.246.102", 6508, uuids1, itemIds1, value, valueTime)

	// // to region mesos node
	// ThirdpardRequest("172.28.196.11", 6508, uuids2, itemIds1, value, valueTime)

	// // to zone => region
	// ThirdpardRequest("172.28.196.84", 6508, uuids3, itemIds1, value, valueTime)

	// // to zone
	// ThirdpardRequest("172.28.196.84", 6508, uuids4, itemIds2, value, valueTime)
}

func ThirdpardRequest(ip string, port int, uuids []string, itemIds []int, value int64, valueTime int64) error {
	info := new(umonitor2.ThirdpartStatsRequest)
	postReq := ufmessage.NewMessage(umonitor2.MessageType_value["THIRDPART_STATS_REQUEST"], "", false, 1, 0, "third part stats")
	for i, _ := range uuids {
		for j, _ := range itemIds {
			thirdpartInfo := new(umonitor2.ThirdpartStatsInfo)
			thirdpartInfo.Uuid = &uuids[i]
			thirdpartInfo.ItemId = proto.Uint32(uint32(itemIds[j]))
			thirdpartInfo.Value = proto.Int64(value)
			thirdpartInfo.Time = proto.Uint32(uint32(valueTime))
			info.ThirdpartStatsList = append(info.ThirdpartStatsList, thirdpartInfo)
		}
	}
	proto.SetExtension(postReq.GetBody(), umonitor2.E_ThirdpartStatsRequest, info)
	fmt.Println(postReq)
	buffer, err := proto.Marshal(postReq)
	if err != nil {
		uflog.ERRORF("[ThirdpartRequest] proto.Marshal failed: %v", err)
		return err
	}
	err = ufnet.SendTCPRequestNoResponse(ip, port, buffer, 20)
	if err != nil {
		uflog.ERRORF("[ThirdpartRequest] ufnet.SendTCPRequestNoResponse failed: %v", err)
		return err
	}
	fmt.Println("request success")
	return nil
}
