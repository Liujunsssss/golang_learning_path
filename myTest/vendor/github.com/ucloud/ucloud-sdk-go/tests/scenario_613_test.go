// Code is generated by ucloud-model, DO NOT EDIT IT.

package tests

import (
	"testing"
	"time"

	"github.com/ucloud/ucloud-sdk-go/services/uhost"
	"github.com/ucloud/ucloud-sdk-go/services/unet"
	"github.com/ucloud/ucloud-sdk-go/services/vpc"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/driver"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/functions"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/utils"
	"github.com/ucloud/ucloud-sdk-go/ucloud/utest/validation"
)

func TestScenario613(t *testing.T) {
	spec.ParallelTest(t, &driver.Scenario{
		PreCheck: func() {
			testAccPreCheck(t)
		},
		Id: "613",
		Vars: func(scenario *driver.Scenario) map[string]interface{} {
			return map[string]interface{}{
				"Image_Id": "#{u_get_image_resource($Region,$Zone)}",
				"Region":   "cn-bj2",
				"Zone":     "cn-bj2-02",
			}
		},
		Owners: []string{"li.wei@ucloud.cn"},
		Title:  "新版NAT网关-natgw自动化回归-端口转发-02-BGP线路",
		Steps: []*driver.Step{
			testStep613DescribeImage01,
			testStep613CreateVPC02,
			testStep613CreateSubnet03,
			testStep613CreateUHostInstance04,
			testStep613AllocateEIP05,
			testStep613DescribeFirewall06,
			testStep613CreateNATGW07,
			testStep613DescribeEIP08,
			testStep613CreateNATGWPolicy09,
			testStep613CreateNATGWPolicy10,
			testStep613CreateNATGWPolicy11,
			testStep613CreateNATGWPolicy12,
			testStep613DescribeNATGWPolicy13,
			testStep613GetAvailableResourceForPolicy14,
			testStep613UpdateNATGWPolicy15,
			testStep613UpdateNATGWPolicy16,
			testStep613UpdateNATGWPolicy17,
			testStep613UpdateNATGWPolicy18,
			testStep613DeleteNATGWPolicy19,
			testStep613DeleteNATGWPolicy20,
			testStep613DeleteNATGWPolicy21,
			testStep613DeleteNATGWPolicy22,
			testStep613DeleteNATGW23,
			testStep613ReleaseEIP24,
			testStep613PoweroffUHostInstance25,
			testStep613TerminateUHostInstance26,
			testStep613DeleteSubnet27,
			testStep613DeleteVPC28,
		},
	})
}

var testStep613DescribeImage01 = &driver.Step{
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

		step.Scenario.SetVar("Image_Id", step.Must(utils.GetValue(resp, "ImageSet.0.ImageId")))
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
	FastFail:      false,
}

var testStep613CreateVPC02 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateVPCRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region": step.Scenario.GetVar("Region"),
			"Network": []interface{}{
				"172.16.0.0/12",
			},
			"Name": "vpc-natgw-bgp",
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateVPC(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("VPCId", step.Must(utils.GetValue(resp, "VPCId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(0) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建VPC",
	FastFail:      false,
}

var testStep613CreateSubnet03 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateSubnetRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"VPCId":      step.Scenario.GetVar("VPCId"),
			"SubnetName": "natgw-s1-bgp",
			"Subnet":     "172.16.0.0",
			"Region":     step.Scenario.GetVar("Region"),
			"Netmask":    21,
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateSubnet(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("SubnetId", step.Must(utils.GetValue(resp, "SubnetId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建子网",
	FastFail:      false,
}

var testStep613CreateUHostInstance04 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewCreateUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":        step.Scenario.GetVar("Zone"),
			"VPCId":       step.Scenario.GetVar("VPCId"),
			"Tag":         "Default",
			"SubnetId":    step.Scenario.GetVar("SubnetId"),
			"Region":      step.Scenario.GetVar("Region"),
			"Password":    "VXFhNzg5VGVzdCFAIyQ7LA==",
			"Name":        "natgw-s1-bgp",
			"Memory":      1024,
			"MachineType": "N",
			"LoginMode":   "Password",
			"ImageId":     step.Scenario.GetVar("Image_Id"),
			"Disks": []map[string]interface{}{
				{
					"IsBoot": "True",
					"Size":   20,
					"Type":   "LOCAL_NORMAL",
				},
			},
			"CPU": 1,
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateUHostInstance(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("UHostIds_s1", step.Must(utils.GetValue(resp, "UHostIds.0")))
		step.Scenario.SetVar("IPs_s1", step.Must(utils.GetValue(resp, "IPs.0")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建云主机",
	FastFail:      false,
}

var testStep613AllocateEIP05 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UNet")
		if err != nil {
			return nil, err
		}
		client := c.(*unet.UNetClient)

		req := client.NewAllocateEIPRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Tag":          "Default",
			"Region":       step.Scenario.GetVar("Region"),
			"Quantity":     1,
			"PayMode":      "Bandwidth",
			"OperatorName": "Bgp",
			"Name":         "natgw-eip-bgp",
			"ChargeType":   "Month",
			"Bandwidth":    2,
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.AllocateEIP(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("EIPId", step.Must(utils.GetValue(resp, "EIPSet.0.EIPId")))
		step.Scenario.SetVar("EIP", step.Must(utils.GetValue(resp, "EIPSet.0.EIPAddr.0.IP")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(180) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "申请弹性IP",
	FastFail:      false,
}

var testStep613DescribeFirewall06 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UNet")
		if err != nil {
			return nil, err
		}
		client := c.(*unet.UNetClient)

		req := client.NewDescribeFirewallRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region": step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeFirewall(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("FWId", step.Must(utils.GetValue(resp, "DataSet")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "获取防火墙信息",
	FastFail:      false,
}

var testStep613CreateNATGW07 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateNATGWRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"VPCId": step.Scenario.GetVar("VPCId"),
			"Tag":   "Default",
			"SubnetworkIds": []interface{}{
				step.Scenario.GetVar("SubnetId"),
			},
			"Remark":     "bgp",
			"Region":     step.Scenario.GetVar("Region"),
			"NATGWName":  "natgw-bgp",
			"FirewallId": step.Must(functions.SearchValue(step.Scenario.GetVar("FWId"), "Type", "recommend web", "FWId")),
			"EIPIds": []interface{}{
				step.Scenario.GetVar("EIPId"),
			},
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateNATGW(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    10,
	RetryInterval: 10 * time.Second,
	Title:         "创建NatGateway",
	FastFail:      false,
}

var testStep613DescribeEIP08 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UNet")
		if err != nil {
			return nil, err
		}
		client := c.(*unet.UNetClient)

		req := client.NewDescribeEIPRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region": step.Scenario.GetVar("Region"),
			"EIPIds": []interface{}{
				step.Scenario.GetVar("EIPId"),
			},
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeEIP(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("NATGWId", step.Must(utils.GetValue(resp, "EIPSet.0.Resource.ResourceID")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
			validation.Builtins.NewValidator("EIPSet.0.Resource.ResourceType", "natgw", "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "获取弹性IP信息",
	FastFail:      false,
}

var testStep613CreateNATGWPolicy09 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    80,
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "TCP",
			"PolicyName": "tcp",
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    80,
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("PolicyId_01", step.Must(utils.GetValue(resp, "PolicyId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613CreateNATGWPolicy10 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    80,
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "UDP",
			"PolicyName": "udp",
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    80,
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("PolicyId_02", step.Must(utils.GetValue(resp, "PolicyId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613CreateNATGWPolicy11 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    "1024-2048",
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "TCP",
			"PolicyName": "tcp段",
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    "1024-2048",
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("PolicyId_03", step.Must(utils.GetValue(resp, "PolicyId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613CreateNATGWPolicy12 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewCreateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    "1024-2048",
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "UDP",
			"PolicyName": "udp段",
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    "1024-2048",
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.CreateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		step.Scenario.SetVar("PolicyId_04", step.Must(utils.GetValue(resp, "PolicyId")))
		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "创建 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613DescribeNATGWPolicy13 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDescribeNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":  step.Scenario.GetVar("Region"),
			"NATGWId": step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DescribeNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "查看 NATGW 端口转发策略",
	FastFail:      false,
}

var testStep613GetAvailableResourceForPolicy14 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewGetAvailableResourceForPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":  step.Scenario.GetVar("Region"),
			"NATGWId": step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.GetAvailableResourceForPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "获得NATGW支持的可端口转发资源信息",
	FastFail:      false,
}

var testStep613UpdateNATGWPolicy15 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewUpdateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    90,
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "UDP",
			"PolicyName": "udp-gai",
			"PolicyId":   step.Scenario.GetVar("PolicyId_01"),
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    90,
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.UpdateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "更新 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613UpdateNATGWPolicy16 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewUpdateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    90,
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "TCP",
			"PolicyName": "tcp-gai",
			"PolicyId":   step.Scenario.GetVar("PolicyId_02"),
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    90,
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.UpdateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "更新 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613UpdateNATGWPolicy17 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewUpdateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    "8080-8088",
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "UDP",
			"PolicyName": "udp段-gai",
			"PolicyId":   step.Scenario.GetVar("PolicyId_03"),
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    "8080-8088",
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.UpdateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "更新 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613UpdateNATGWPolicy18 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewUpdateNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SrcPort":    "8080-8088",
			"SrcEIPId":   step.Scenario.GetVar("EIPId"),
			"Region":     step.Scenario.GetVar("Region"),
			"Protocol":   "TCP",
			"PolicyName": "tcp段-gai",
			"PolicyId":   step.Scenario.GetVar("PolicyId_04"),
			"NATGWId":    step.Scenario.GetVar("NATGWId"),
			"DstPort":    "8080-8088",
			"DstIP":      step.Scenario.GetVar("IPs_s1"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.UpdateNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "更新 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613DeleteNATGWPolicy19 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":   step.Scenario.GetVar("Region"),
			"PolicyId": step.Scenario.GetVar("PolicyId_01"),
			"NATGWId":  step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613DeleteNATGWPolicy20 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":   step.Scenario.GetVar("Region"),
			"PolicyId": step.Scenario.GetVar("PolicyId_02"),
			"NATGWId":  step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613DeleteNATGWPolicy21 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":   step.Scenario.GetVar("Region"),
			"PolicyId": step.Scenario.GetVar("PolicyId_03"),
			"NATGWId":  step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613DeleteNATGWPolicy22 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteNATGWPolicyRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":   step.Scenario.GetVar("Region"),
			"PolicyId": step.Scenario.GetVar("PolicyId_04"),
			"NATGWId":  step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteNATGWPolicy(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除 NATGateWay 转发策略",
	FastFail:      false,
}

var testStep613DeleteNATGW23 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteNATGWRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region":  step.Scenario.GetVar("Region"),
			"NATGWId": step.Scenario.GetVar("NATGWId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteNATGW(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{
			validation.Builtins.NewValidator("RetCode", 0, "str_eq"),
		}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除NatGateway",
	FastFail:      false,
}

var testStep613ReleaseEIP24 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UNet")
		if err != nil {
			return nil, err
		}
		client := c.(*unet.UNetClient)

		req := client.NewReleaseEIPRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Region": step.Scenario.GetVar("Region"),
			"EIPId":  step.Scenario.GetVar("EIPId"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.ReleaseEIP(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{}
	},
	StartupDelay:  time.Duration(3) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "释放弹性IP",
	FastFail:      false,
}

var testStep613PoweroffUHostInstance25 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewPoweroffUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":    step.Scenario.GetVar("Zone"),
			"UHostId": step.Scenario.GetVar("UHostIds_s1"),
			"Region":  step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.PoweroffUHostInstance(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{}
	},
	StartupDelay:  time.Duration(5) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "模拟主机掉电",
	FastFail:      false,
}

var testStep613TerminateUHostInstance26 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("UHost")
		if err != nil {
			return nil, err
		}
		client := c.(*uhost.UHostClient)

		req := client.NewTerminateUHostInstanceRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"Zone":    step.Scenario.GetVar("Zone"),
			"UHostId": step.Scenario.GetVar("UHostIds_s1"),
			"Region":  step.Scenario.GetVar("Region"),
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
		return []driver.TestValidator{}
	},
	StartupDelay:  time.Duration(60) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除云主机",
	FastFail:      false,
}

var testStep613DeleteSubnet27 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteSubnetRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"SubnetId": step.Scenario.GetVar("SubnetId"),
			"Region":   step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteSubnet(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{}
	},
	StartupDelay:  time.Duration(5) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除子网",
	FastFail:      false,
}

var testStep613DeleteVPC28 = &driver.Step{
	Invoker: func(step *driver.Step) (interface{}, error) {
		c, err := step.LoadFixture("VPC")
		if err != nil {
			return nil, err
		}
		client := c.(*vpc.VPCClient)

		req := client.NewDeleteVPCRequest()
		err = utils.SetRequest(req, map[string]interface{}{
			"VPCId":  step.Scenario.GetVar("VPCId"),
			"Region": step.Scenario.GetVar("Region"),
		})
		if err != nil {
			return nil, err
		}

		resp, err := client.DeleteVPC(req)
		if err != nil {
			return resp, err
		}

		return resp, nil
	},
	Validators: func(step *driver.Step) []driver.TestValidator {
		return []driver.TestValidator{}
	},
	StartupDelay:  time.Duration(5) * time.Second,
	MaxRetries:    3,
	RetryInterval: 10 * time.Second,
	Title:         "删除VPC",
	FastFail:      false,
}
