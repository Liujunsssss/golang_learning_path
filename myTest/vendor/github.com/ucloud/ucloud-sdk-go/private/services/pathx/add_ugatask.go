//Code is generated by ucloud code generator, don't modify it by hand, it will cause undefined behaviors.
//go:generate ucloud-gen-go-api PathX AddUGATask

package pathx

import (
	"github.com/ucloud/ucloud-sdk-go/ucloud/request"
	"github.com/ucloud/ucloud-sdk-go/ucloud/response"
)

// AddUGATaskRequest is request schema for AddUGATask action
type AddUGATaskRequest struct {
	request.CommonBase

	// [公共参数] 项目ID。请参考[GetProjectList接口](../summary/get_project_list.html)
	// ProjectId *string `required:"true"`

	// 全球加速实例ID
	UGAId *string `required:"true"`

	// TCP端口号
	TCP []string `required:"false"`

	// UDP端口号
	UDP []string `required:"false"`

	// HTTP端口号
	HTTP []string `required:"false"`

	// HTTPS端口号
	HTTPS []string `required:"false"`
}

// AddUGATaskResponse is response schema for AddUGATask action
type AddUGATaskResponse struct {
	response.CommonBase
}

// NewAddUGATaskRequest will create request of AddUGATask action.
func (c *PathXClient) NewAddUGATaskRequest() *AddUGATaskRequest {
	req := &AddUGATaskRequest{}

	// setup request with client config
	c.Client.SetupRequest(req)

	// setup retryable with default retry policy (retry for non-create action and common error)
	req.SetRetryable(false)
	return req
}

// AddUGATask - 添加加速配置端口
func (c *PathXClient) AddUGATask(req *AddUGATaskRequest) (*AddUGATaskResponse, error) {
	var err error
	var res AddUGATaskResponse

	err = c.Client.InvokeAction("AddUGATask", req, &res)
	if err != nil {
		return &res, err
	}

	return &res, nil
}
