// Code is generated by ucloud-model, DO NOT EDIT IT.

package tests

import (
	"testing"
	"time"

	"github.com/ucloud/ucloud-sdk-go/services/ubill"
	"github.com/ucloud/ucloud-sdk-go/services/udisk"
	"github.com/ucloud/ucloud-sdk-go/services/uhost"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/driver"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/functions"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/utils"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/validation"
)

func TestScenario3299(t *testing.T) {
	spec.ParallelTest(t, &driver.Scenario{
		PreCheck: func() {
			testAccPreCheck(t)
		},
		Id: "3299",
		Vars: func(scenario *driver.Scenario) map[string]interface{} {
			return map[string]interface{}{
				"DiskSpace":          30,
				"BackupMode":         "DATAARK",
				"NormalDiskSpace":    20,
				"ImageID":            "#{u_get_image_resource($Region,$Zone)}",
				"Disk0Type":          "CLOUD_SSD",
				"Disk1Type":          "CLOUD_NORMAL",
				"Disk0Backup":        "NONE",
				"Disk1Backup":        "NONE",
				"Password":           "dXFhI3VjbG91ZC5jbiFA",
				"MinimalCpuPlatform": "Intel/Auto",
				"MachineType":        "N",
				"CreateCPU":          4,
				"CreateMem":          8192,
				"ChargeType":         "Month",
				"Disk0Size":          20,
				"Disk1Size":          20,
				"Region":             "cn-bj2",
				"Zone":               "cn-bj2-05",
			}
		},
		Owners: []string{"maggie.an@ucloud.cn"},
		Title:  "N云盘主机-无方舟-方舟",
		Steps: []*driver.Step{
			testStep3299DescribeImage01,
			testStep3299GetUHostInstancePrice02,
			testStep3299CreateUHostInstance03,
			testStep3299DescribeUHostInstance04,
			testStep3299DescribeOrderDetailInfo05,
			testStep3299DescribeOrderDetailInfo06,
			testStep3299StopUHostInstance07,
			testStep3299DescribeUHostInstance08,
			testStep3299GetAttachedDiskUpgradePrice09,
			testStep3299SetUDiskUDataArkMode10,
			testStep3299DescribeOrderDetailInfo11,
			testStep3299DescribeUHostInstance12,
			testStep3299GetAttachedDiskUpgradePrice13,
			testStep3299ResizeAttachedDisk14,
			testStep3299DescribeOrderDetailInfo15,
			testStep3299DescribeUHostInstance16,
			testStep3299TerminateUHostInstance17,
		},
	})
}

var testStep3299DescribeImage01 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewDescribeImageRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":      step.Scenario.GetVar("Zone"),
			"Region":    step.Scenario.GetVar("Region"),
			"OsType":    "Linux",
			"ImageType": "Base",
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeImage(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("ImageID", step.Must(utils.GetValue(resp, "ImageSet.0.ImageId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeImageResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取镜像列表",
	FastFail:      true,
}

var testStep3299GetUHostInstancePrice02 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewGetUHostInstancePriceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":        step.Scenario.GetVar("Zone"),
			"Region":      step.Scenario.GetVar("Region"),
			"Memory":      step.Scenario.GetVar("CreateMem"),
			"MachineType": step.Scenario.GetVar("MachineType"),
			"ImageId":     step.Scenario.GetVar("ImageID"),
			"Disks": []map[string]interface{}{
				{
					"BackupType": step.Scenario.GetVar("Disk0Backup"),
					"IsBoot":     "True",
					"Size":       step.Scenario.GetVar("Disk0Size"),
					"Type":       step.Scenario.GetVar("Disk0Type"),
				},
				{
					"BackupType": step.Scenario.GetVar("Disk1Backup"),
					"IsBoot":     "False",
					"Size":       step.Scenario.GetVar("Disk1Size"),
					"Type":       step.Scenario.GetVar("Disk1Type"),
				},
			},
			"Count":      1,
			"ChargeType": step.Scenario.GetVar("ChargeType"),
			"CPU":        step.Scenario.GetVar("CreateCPU"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetUHostInstancePrice(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "GetUHostInstancePriceResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取主机价格",
	FastFail:      true,
}

var testStep3299CreateUHostInstance03 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewCreateUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":               step.Scenario.GetVar("Zone"),
			"Region":             step.Scenario.GetVar("Region"),
			"Password":           "VXFhNzg5VGVzdCFAIyQ7LA==",
			"Name":               "SSD云-普通云-无方舟-方舟",
			"MinimalCpuPlatform": step.Scenario.GetVar("MinimalCpuPlatform"),
			"Memory":             step.Scenario.GetVar("CreateMem"),
			"MachineType":        step.Scenario.GetVar("MachineType"),
			"LoginMode":          "Password",
			"ImageId":            step.Scenario.GetVar("ImageID"),
			"Disks": []map[string]interface{}{
				{
					"BackupType": step.Scenario.GetVar("Disk0Backup"),
					"IsBoot":     "True",
					"Size":       20,
					"Type":       step.Scenario.GetVar("Disk0Type"),
				},
				{
					"BackupType": step.Scenario.GetVar("Disk1Backup"),
					"IsBoot":     "False",
					"Size":       20,
					"Type":       step.Scenario.GetVar("Disk1Type"),
				},
			},
			"ChargeType": step.Scenario.GetVar("ChargeType"),
			"CPU":        step.Scenario.GetVar("CreateCPU"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateUHostInstance(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("UHostId1", step.Must(utils.GetValue(resp, "UHostIds.0")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "CreateUHostInstanceResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "创建云主机",
	FastFail:      true,
}

var testStep3299DescribeUHostInstance04 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewDescribeUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone": step.Scenario.GetVar("Zone"),
			"UHostIds": []interface{}{
				step.Scenario.GetVar("UHostId1"),
			},
			"Region": step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeUHostInstance(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("DiskIdSys1", step.Must(utils.GetValue(resp, "UHostSet.0.DiskSet.0.DiskId")))
		step.Scenario.SetVar("DiskIdData1", step.Must(utils.GetValue(resp, "UHostSet.0.DiskSet.1.DiskId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeUHostInstanceResponse", "str_eq"),
			validation.Builtins.NewValidator("UHostSet.0.State", "Running", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    30,
	RetryInterval: 30 * time.Second,
	Title:         "获取主机信息",
	FastFail:      true,
}

var testStep3299DescribeOrderDetailInfo05 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UBill")
		if err != nil {
			return nil, err
		}
		client := c.(*ubill.UBillClient)

		req := client.NewDescribeOrderDetailInfoRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"ResourceIds": []interface{}{
				step.Scenario.GetVar("UHostId1"),
			},
			"OrderTypes": []interface{}{
				"OT_BUY",
			},
			"EndTime":   step.Must(functions.GetTimestamp(10)),
			"BeginTime": step.Must(functions.Calculate("-", step.Must(functions.GetTimestamp(10)), 500)),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeOrderDetailInfo(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeOrderDetailInfoResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取订单信息",
	FastFail:      true,
}

var testStep3299DescribeOrderDetailInfo06 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UBill")
		if err != nil {
			return nil, err
		}
		client := c.(*ubill.UBillClient)

		req := client.NewDescribeOrderDetailInfoRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"ResourceIds": []interface{}{
				step.Scenario.GetVar("DiskIdData1"),
			},
			"OrderTypes": []interface{}{
				"OT_BUY",
			},
			"EndTime":   step.Must(functions.GetTimestamp(10)),
			"BeginTime": step.Must(functions.Calculate("-", step.Must(functions.GetTimestamp(10)), 500)),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeOrderDetailInfo(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeOrderDetailInfoResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取订单信息",
	FastFail:      true,
}

var testStep3299StopUHostInstance07 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewStopUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":    step.Scenario.GetVar("Zone"),
			"UHostId": step.Scenario.GetVar("UHostId1"),
			"Region":  step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.StopUHostInstance(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "StopUHostInstanceResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "关闭主机",
	FastFail:      true,
}

var testStep3299DescribeUHostInstance08 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewDescribeUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone": step.Scenario.GetVar("Zone"),
			"UHostIds": []interface{}{
				step.Scenario.GetVar("UHostId1"),
			},
			"Region": step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeUHostInstance(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeUHostInstanceResponse", "str_eq"),
			validation.Builtins.NewValidator("UHostSet.0.State", "Stopped", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    30,
	RetryInterval: 10 * time.Second,
	Title:         "获取主机信息",
	FastFail:      true,
}

var testStep3299GetAttachedDiskUpgradePrice09 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewGetAttachedDiskUpgradePriceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":       step.Scenario.GetVar("Zone"),
			"UHostId":    step.Scenario.GetVar("UHostId1"),
			"Region":     step.Scenario.GetVar("Region"),
			"DiskSpace":  step.Scenario.GetVar("NormalDiskSpace"),
			"DiskId":     step.Scenario.GetVar("DiskIdData1"),
			"BackupMode": step.Scenario.GetVar("BackupMode"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetAttachedDiskUpgradePrice(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("Price2", step.Must(utils.GetValue(resp, "Price")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "GetAttachedDiskUpgradePriceResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取挂载磁盘的升级价格",
	FastFail:      true,
}

var testStep3299SetUDiskUDataArkMode10 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UDisk")
		if err != nil {
			return nil, err
		}
		client := c.(*udisk.UDiskClient)

		req := client.NewSetUDiskUDataArkModeRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":         step.Scenario.GetVar("Zone"),
			"UDiskId":      step.Scenario.GetVar("DiskIdData1"),
			"UDataArkMode": "Yes",
			"Region":       step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.SetUDiskUDataArkMode(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "SetUDiskUDataArkModeResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "设置UDisk数据方舟的状态",
	FastFail:      true,
}

var testStep3299DescribeOrderDetailInfo11 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UBill")
		if err != nil {
			return nil, err
		}
		client := c.(*ubill.UBillClient)

		req := client.NewDescribeOrderDetailInfoRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"ResourceIds": []interface{}{
				step.Scenario.GetVar("DiskIdData1"),
			},
			"OrderTypes": []interface{}{
				"OT_UPGRADE",
			},
			"EndTime":   step.Must(functions.GetTimestamp(10)),
			"BeginTime": step.Must(functions.Calculate("-", step.Must(functions.GetTimestamp(10)), 100)),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeOrderDetailInfo(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeOrderDetailInfoResponse", "str_eq"),
			validation.Builtins.NewValidator("OrderInfos.0.Amount", step.Must(functions.Calculate("*", step.Scenario.GetVar("Price2"), 0.99)), "gt"),
			validation.Builtins.NewValidator("OrderInfos.0.Amount", step.Must(functions.Calculate("*", step.Scenario.GetVar("Price2"), 1.01)), "lt"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取订单信息",
	FastFail:      true,
}

var testStep3299DescribeUHostInstance12 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewDescribeUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone": step.Scenario.GetVar("Zone"),
			"UHostIds": []interface{}{
				step.Scenario.GetVar("UHostId1"),
			},
			"Region": step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeUHostInstance(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeUHostInstanceResponse", "str_eq"),
			validation.Builtins.NewValidator("UHostSet.0.State", "Stopped", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    30,
	RetryInterval: 30 * time.Second,
	Title:         "获取主机信息",
	FastFail:      true,
}

var testStep3299GetAttachedDiskUpgradePrice13 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewGetAttachedDiskUpgradePriceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":       step.Scenario.GetVar("Zone"),
			"UHostId":    step.Scenario.GetVar("UHostId1"),
			"Region":     step.Scenario.GetVar("Region"),
			"DiskSpace":  step.Scenario.GetVar("DiskSpace"),
			"DiskId":     step.Scenario.GetVar("DiskIdData1"),
			"BackupMode": step.Scenario.GetVar("BackupMode"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetAttachedDiskUpgradePrice(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("Price3", step.Must(utils.GetValue(resp, "Price")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "GetAttachedDiskUpgradePriceResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取挂载磁盘的升级价格",
	FastFail:      true,
}

var testStep3299ResizeAttachedDisk14 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewResizeAttachedDiskRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":      step.Scenario.GetVar("Zone"),
			"UHostId":   step.Scenario.GetVar("UHostId1"),
			"Region":    step.Scenario.GetVar("Region"),
			"DiskSpace": step.Scenario.GetVar("DiskSpace"),
			"DiskId":    step.Scenario.GetVar("DiskIdData1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.ResizeAttachedDisk(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "ResizeAttachedDiskResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(60) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "修改挂载的磁盘大小",
	FastFail:      true,
}

var testStep3299DescribeOrderDetailInfo15 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UBill")
		if err != nil {
			return nil, err
		}
		client := c.(*ubill.UBillClient)

		req := client.NewDescribeOrderDetailInfoRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"ResourceIds": []interface{}{
				step.Scenario.GetVar("DiskIdData1"),
			},
			"OrderTypes": []interface{}{
				"OT_UPGRADE",
			},
			"EndTime":   step.Must(functions.GetTimestamp(10)),
			"BeginTime": step.Must(functions.Calculate("-", step.Must(functions.GetTimestamp(10)), 100)),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeOrderDetailInfo(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeOrderDetailInfoResponse", "str_eq"),
			validation.Builtins.NewValidator("OrderInfos.0.Amount", step.Must(functions.Calculate("*", step.Scenario.GetVar("Price3"), 0.99)), "gt"),
			validation.Builtins.NewValidator("OrderInfos.0.Amount", step.Must(functions.Calculate("*", step.Scenario.GetVar("Price3"), 1.01)), "lt"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "获取订单信息",
	FastFail:      true,
}

var testStep3299DescribeUHostInstance16 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewDescribeUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone": step.Scenario.GetVar("Zone"),
			"UHostIds": []interface{}{
				step.Scenario.GetVar("UHostId1"),
			},
			"Region": step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeUHostInstance(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "DescribeUHostInstanceResponse", "str_eq"),
			validation.Builtins.NewValidator("UHostSet.0.State", "Stopped", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    30,
	RetryInterval: 30 * time.Second,
	Title:         "获取主机信息",
	FastFail:      true,
}

var testStep3299TerminateUHostInstance17 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewTerminateUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":         step.Scenario.GetVar("Zone"),
			"UHostId":      step.Scenario.GetVar("UHostId1"),
			"ReleaseUDisk": "true",
			"Region":       step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.TerminateUHostInstance(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("Action", "TerminateUHostInstanceResponse", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 1 * time.Second,
	Title:         "删除云主机",
	FastFail:      true,
}
