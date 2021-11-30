// Code is generated by ucloud-model, DO NOT EDIT IT.

package tests

import (
	"testing"
	"time"

	"github.com/ucloud/ucloud-sdk-go/services/udts"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/driver"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/utils"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/validation"
)

func TestScenario5081(t *testing.T) {
	spec.ParallelTest(t, &driver.Scenario{
		PreCheck: func() {
			testAccPreCheck(t)
		},
		Id: "5081",
		Vars: func(scenario *driver.Scenario) map[string]interface{} {
			return map[string]interface{}{}
		},
		Owners: []string{"chenoa.chen@ucloud.cn"},
		Title:  " UDTS-官网API测试集",
		Steps: []*driver.Step{
			testStep5081ListUDTSTask01,
			testStep5081GetUDTSTaskConfigure02,
			testStep5081StartUDTSTask03,
			testStep5081StopUDTSTask04,
			testStep5081GetUDTSTaskStatus05,
			testStep5081StartUDTSTask06,
			testStep5081GetUDTSTaskStatus07,
		},
	})
}

var testStep5081ListUDTSTask01 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewListUDTSTaskRequest()
		err = utils.SetRequest(req, map[string]interface{}{})
		if err != nil {
			return nil, err
		}

		resp, err := client.ListUDTSTask(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("TaskId", step.Must(utils.GetValue(resp, "Data.0.TaskId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "ListUDTSTaskResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取 Task 列表信息",
	FastFail:      false,
}

var testStep5081GetUDTSTaskConfigure02 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewGetUDTSTaskConfigureRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"TaskId": step.Scenario.GetVar("TaskId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetUDTSTaskConfigure(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "GetUDTSTaskConfigureResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取任务配置",
	FastFail:      false,
}

var testStep5081StartUDTSTask03 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewStartUDTSTaskRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"TaskId": step.Scenario.GetVar("TaskId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.StartUDTSTask(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "StartUDTSTaskResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "启动UDTS服务",
	FastFail:      false,
}

var testStep5081StopUDTSTask04 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewStopUDTSTaskRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"TaskId": step.Scenario.GetVar("TaskId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.StopUDTSTask(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "StopUDTSTaskResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "停止UDTS任务",
	FastFail:      false,
}

var testStep5081GetUDTSTaskStatus05 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewGetUDTSTaskStatusRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"TaskId": step.Scenario.GetVar("TaskId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetUDTSTaskStatus(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "GetUDTSTaskStatusResponse", "str_eq"),
			validation.Builtins.NewValidator("Data.Status", "Stopped", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "查看服务状态",
	FastFail:      false,
}

var testStep5081StartUDTSTask06 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewStartUDTSTaskRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"TaskId": step.Scenario.GetVar("TaskId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.StartUDTSTask(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "StartUDTSTaskResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "启动UDTS服务",
	FastFail:      false,
}

var testStep5081GetUDTSTaskStatus07 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDTS")
		if err != nil {
			return nil, err
		}
		client := c.(*udts.UDTSClient)

		req := client.NewGetUDTSTaskStatusRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"TaskId": step.Scenario.GetVar("TaskId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetUDTSTaskStatus(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "GetUDTSTaskStatusResponse", "str_eq"),
			validation.Builtins.NewValidator("Data.Status", "Done", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    300,
	RetryInterval: 2 * time.Second,
	Title:         "查看服务状态",
	FastFail:      false,
}
