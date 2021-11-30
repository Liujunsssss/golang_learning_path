//Code is generated by ucloud code generator, don't modify it by hand, it will cause undefined behaviors.
//go:generate ucloud-gen-go-api UDisk DescribeUDisk

package udisk

import (
	"github.com/ucloud/ucloud-sdk-go/ucloud/request"
	"github.com/ucloud/ucloud-sdk-go/ucloud/response"
)

// DescribeUDiskRequest is request schema for DescribeUDisk action
type DescribeUDiskRequest struct {
	request.CommonBase

	// [公共参数] 地域。 参见 [地域和可用区列表](../summary/regionlist.html)
	// Region *string `required:"true"`

	// [公共参数] 可用区。参见 [可用区列表](../summary/regionlist.html)
	// Zone *string `required:"false"`

	// [公共参数] 项目ID。不填写为默认项目，子帐号必须填写。 请参考[GetProjectList接口](../summary/get_project_list.html)
	// ProjectId *string `required:"false"`

	// UDisk Id(留空返回全部)
	UDiskId *string `required:"false"`

	// 数据偏移量, 默认为0
	Offset *int `required:"false"`

	// 返回数据长度, 默认为20
	Limit *int `required:"false"`

	// 普通数据盘:DataDisk; 普通系统盘:SystemDisk; SSD数据盘:SSDDataDisk; RSSD数据盘:RSSDDataDisk; 为空拉取所有
	DiskType *string `required:"false"`
}

// DescribeUDiskResponse is response schema for DescribeUDisk action
type DescribeUDiskResponse struct {
	response.CommonBase

	// JSON 格式的UDisk数据列表, 每项参数可见下面 UDiskDataSet
	DataSet []UDiskDataSet

	// 根据过滤条件得到的总数
	TotalCount int
}

// NewDescribeUDiskRequest will create request of DescribeUDisk action.
func (c *UDiskClient) NewDescribeUDiskRequest() *DescribeUDiskRequest {
	req := &DescribeUDiskRequest{}

	// setup request with client config
	c.Client.SetupRequest(req)

	// setup retryable with default retry policy (retry for non-create action and common error)
	req.SetRetryable(true)
	return req
}

// DescribeUDisk - 获取UDisk实例
func (c *UDiskClient) DescribeUDisk(req *DescribeUDiskRequest) (*DescribeUDiskResponse, error) {
	var err error
	var res DescribeUDiskResponse

	err = c.Client.InvokeAction("DescribeUDisk", req, &res)
	if err != nil {
		return &res, err
	}

	return &res, nil
}
