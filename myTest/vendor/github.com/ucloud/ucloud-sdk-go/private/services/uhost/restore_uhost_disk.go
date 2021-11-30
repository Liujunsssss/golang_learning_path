//Code is generated by ucloud code generator, don't modify it by hand, it will cause undefined behaviors.
//go:generate ucloud-gen-go-api UHost RestoreUHostDisk

package uhost

import (
	"github.com/ucloud/ucloud-sdk-go/ucloud/request"
	"github.com/ucloud/ucloud-sdk-go/ucloud/response"
)

// RestoreUHostDiskRequest is request schema for RestoreUHostDisk action
type RestoreUHostDiskRequest struct {
	request.CommonBase

	// 可用区。参见 [可用区列表](../summary/regionlist.html)
	// Zone *string `required:"true"`

	// 快照所属主机.仅当网盘数据盘未挂载时才可以不传。
	UHostId *string `required:"false"`

	// 主机上要恢复的磁盘Id. 对于本地盘主机，支持单独恢复系统盘或数据盘，也可以同时恢复系统盘和数据盘；对于网盘主机，只能恢复系统盘。从数据方舟恢复磁盘时必传。
	DiskIds []string `required:"false"`

	// 恢复的盘的时间戳，顺序与DiskIds.n保持对应。从数据方舟恢复磁盘时必传。
	RestoreTimestamps []string `required:"false"`

	// 快照Id. 对于本地盘主机，支持单独恢复系统盘或数据盘，也可以同时恢复系统盘和数据盘；对于网盘主机，只能恢复系统盘。从快照恢复磁盘时必传。
	SnapshotIds []string `required:"false"`
}

// RestoreUHostDiskResponse is response schema for RestoreUHostDisk action
type RestoreUHostDiskResponse struct {
	response.CommonBase
}

// NewRestoreUHostDiskRequest will create request of RestoreUHostDisk action.
func (c *UHostClient) NewRestoreUHostDiskRequest() *RestoreUHostDiskRequest {
	req := &RestoreUHostDiskRequest{}

	// setup request with client config
	c.Client.SetupRequest(req)

	// setup retryable with default retry policy (retry for non-create action and common error)
	req.SetRetryable(false)
	return req
}

// RestoreUHostDisk - 从数据方舟或者快照，恢复主机的磁盘。必须在关机状态下进行。
func (c *UHostClient) RestoreUHostDisk(req *RestoreUHostDiskRequest) (*RestoreUHostDiskResponse, error) {
	var err error
	var res RestoreUHostDiskResponse

	err = c.Client.InvokeAction("RestoreUHostDisk", req, &res)
	if err != nil {
		return &res, err
	}

	return &res, nil
}
