package main

import (
	"fmt"
	"net"
	"strings"
)


curl -vo /dev/null "https://looper015.ucloud.com.cn/conf_agent_redis_1.2.5.10.tar.gz" --resolve "looper015.ucloud.com.cn:443:${1.31.130.65}"

func GetLocalIP() (ip string, err error) {
	ip = "127.0.0.1"
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return
	}

	if conn != nil {
		addr := conn.LocalAddr().String()
		fmt.Println(addr)
		idx := strings.LastIndex(addr, ":")
		if idx == -1 {
			return
		}
		ip = addr[:idx]
	}
	return
}
func main() {
	fmt.Println(GetLocalIP())
}

/*
func a() {
	for {
		time.Sleep(10 * time.Second)
		fmt.Println("wqeqweqweqw")
		time.Sleep(10 * time.Second)
	}
}
func b(m *sync.RWMutex) {
	for {
		m.Lock()
		fmt.Println("11111111111")
		time.Sleep(1 * time.Second)
		m.Unlock()
	}
}
func c(m *sync.RWMutex) {
	for {
		m.Lock()
		fmt.Println("9999999999999999999999")
		time.Sleep(3 * time.Second)
		m.Unlock()
	}
}
func main() {
	m := &sync.RWMutex{}
	go a()
	go b(m)
	go c(m)
	for {
	}
}

/*
func main() {
	for {
		begin := time.Now().Unix()
		host := "http://conf-server.ucloudnaming.com:9999/"
		ConfirmVersionInfoList := make([]map[string]interface{}, 0)
		reqData := map[string]interface{}{
			"Action":                 "ReportVersion",
			"AgentIp":                "1.1.1.1",
			"ProtocolVersion":        2,
			"MaxReceiveVersion":      10,
			"ConfirmVersionInfoList": ConfirmVersionInfoList,
		}
		result := make(map[string]interface{})
		code, err := SimpleJsonHttpPost(host, reqData, &result)
		if err != nil {
			fmt.Println(code)
			fmt.Println(err)
			return
		}
		end := time.Now().Unix()
		fmt.Println(end - begin)
		fmt.Println(code)
		fmt.Println(result)
		time.Sleep(30)
	}
}
func SimpleJsonHttpPost(host string, data interface{}, result interface{}) (code int, err error) {
	jsonStr, _ := json.Marshal(data)

	body, code, err := SimpleHttpPostRequestWitchTimeout(host, "application/json", bytes.NewReader(jsonStr), 20)
	if err != nil {
		return
	}

	err = json.Unmarshal(body, result)
	if err != nil {
		return
	}
	return
}
func SimpleHttpPostRequestWitchTimeout(url string, bodyType string, body io.Reader, timeOut uint32) (res []byte, code int, err error) {
	ctx := context.Background()
	if timeOut != 0 {
		var cancelf func()
		ctx, cancelf = context.WithTimeout(ctx, time.Duration(timeOut)*time.Second)
		defer cancelf()
	}

	rqst, err := http.NewRequest("POST", url, body)
	if err != nil {
		return
	}

	rqst = rqst.WithContext(ctx)
	rqst.Header.Add("Content-Type", bodyType)
	resp, err := http.DefaultClient.Do(rqst)
	if err != nil {
		return
	}

	code = resp.StatusCode

	defer resp.Body.Close()
	res, err = ioutil.ReadAll(resp.Body)
	return
}

/*
func getSignature(params map[string]interface{}) string {
	urlValues := url.Values{}
	for k, v := range params {
		urlValues.Set(k, v.(string))
	}
	cred := &auth.Credential{
		PublicKey:  "H6IfjPpzXlI89GZPTSRtae1mDleW8XBIHLpsZTj9I",
		PrivateKey: "EJWlnSppNUtL5EezekLJ728xhDWolli1D9sHWgPPSVZwFTe211m9JaK6PPJkwYcjMp",
	}
	v := cred.CreateSign(urlValues.Encode())
	return v
}

func main() {
	val := map[string]interface{}{
		"Action":              "AddCertificate",
		"CertName":            "ucloud",
		"top_organization_id": 56032177,
		"organization_id":     63865029,
	}
	fmt.Println(getSignature(val))
}

/*
func main() {
	ch := make(chan int)
	go func() {
		select {
		case ch <- 0:
		case ch <- 1:
		}
	}()

	for v := range ch {
		fmt.Println(v)
	}
}

/*
var m *sync.RWMutex

func main() {
	m = &sync.RWMutex{}
	var n int = 0
	go func() {
		m.Lock()
		n++
		fmt.Println(n)
		m.Unlock()
	}()
	go func() {
		m.Lock()
		n++
		fmt.Println(n)
		m.Unlock()
	}()
	go func() {
		fmt.Println("dsadas")
	}()
	for {
	}
}

/*
//阻塞式的执行外部shell命令的函数,等待执行完毕并返回标准输出
func exec_shell() (string, error) {
	//函数返回一个*Cmd，用于使用给出的参数执行name指定的程序
	cmd := exec.Command("/bin/bash", "-c", "./jsTest.js")

	//读取io.Writer类型的cmd.Stdout，再通过bytes.Buffer(缓冲byte类型的缓冲器)将byte类型转化为string类型(out.String():这是bytes类型提供的接口)
	var out bytes.Buffer
	cmd.Stdout = &out

	//Run执行c包含的命令，并阻塞直到完成。  这里stdout被取出，cmd.Wait()无法正确获取stdin,stdout,stderr，则阻塞在那了
	err := cmd.Run()

	return out.String(), err
}
func main() {
	_, err := exec_shell()
	//fmt.Println(s)
	s := strings.Split(err.Error(), " ")
	fmt.Println(s[2])
	fmt.Println([]byte(err.Error()))
	fmt.Println(err.Error()[len(err.Error())-3:])
}

/*
func GenerateNatural() chan int {
	ch := make(chan int)
	go func() {
		for i := 2; ; i++ {
			ch <- i
		}
	}()
	return ch
}

// 管道过滤器: 删除能被素数整除的数
func PrimeFilter(in <-chan int, prime int) chan int {
	out := make(chan int)
	go func() {
		for {
			i := <-in
			fmt.Printf("%d,%d\n", i, prime)
			if i%prime != 0 {
				out <- i
			}
		}
	}()
	return out
}
func main() {
	ch := GenerateNatural() // 自然数序列: 2, 3, 4, ...
	for i := 0; i < 10; i++ {
		prime := <-ch // 新出现的素数
		//fmt.Printf("%v: %v\n", i+1, prime)
		ch = PrimeFilter(ch, prime) // 基于新素数构造的过滤器
	}
}

/*
type Flock struct {
	LockFile string
	lock     *os.File
}

// 创建文件锁，配合 defer f.Release() 来使用
func Create(file string) (f *Flock, e error) {
	if file == "" {
		e = errors.New("cannot create flock on empty path")
		return
	}
	lock, e := os.Create(file)
	if e != nil {
		return
	}
	return &Flock{
		LockFile: file,
		lock:     lock,
	}, nil
}

// 释放文件锁
func (f *Flock) Release() {
	if f != nil && f.lock != nil {
		f.lock.Close()
		os.Remove(f.LockFile)
	}
}

// 上锁，配合 defer f.Unlock() 来使用
func (f *Flock) Lock() (e error) {
	if f == nil {
		e = errors.New("cannot use lock on a nil flock")
		return
	}
	return syscall.Flock(int(f.lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}

// 解锁
func (f *Flock) Unlock() {
	if f != nil {
		syscall.Flock(int(f.lock.Fd()), syscall.LOCK_UN)
	}
}
func main() {
	lock, e := Create("./myTest")
	if e != nil {
		// handle error
	}
	defer lock.Release()

	// 尝试独占文件锁
	e = lock.Lock()
	if e != nil {
		fmt.Println("sotop")
		// handle error
	}
	for {
	}
	defer lock.Unlock()
}

/*
const SockAddr = "./echo.sock"

func main() {

	l, err := net.Listen("unix", SockAddr)
	if err != nil {
		log.Fatal("listen error:", err)
	}
	defer l.Close()

	for {
		// Accept new connections, dispatching them to echoServer
		// in a goroutine.
		_, err := l.Accept()
		if err != nil {
			log.Fatal("accept error:", err)
		}
	}
}

/*
func main() {
	var unixAddr *net.UnixAddr
	unixAddr, _ = net.ResolveUnixAddr("unix", "./unix")
	unixListener, _ := net.ListenUnix("unix", unixAddr)
	defer unixListener.Close()
	for {
		unixConn, err := unixListener.AcceptUnix()
		if err != nil {
			break
		}
		fmt.Println("A client connected : " + unixConn.RemoteAddr().String())
	}
}

/*
func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func(cancelFunc context.CancelFunc) {
		time.Sleep(10 * time.Second)
		cancelFunc()
	}(cancel)
	Command(ctx, "ping www.baidu.com")
}
func Command(ctx context.Context, cmd string) error {
	c := exec.CommandContext(ctx, "bash", "-c", cmd)
	stdout, err := c.StdoutPipe()
	if err != nil {
		return err
	}
	go func() {
		reader := bufio.NewReader(stdout)
		for {
			// 其实这段去掉程序也会正常运行，只是我们就不知道到底什么时候Command被停止了，而且如果我们需要实时给web端展示输出的话，这里可以作为依据 取消展示
			select {
			// 检测到ctx.Done()之后停止读取
			case <-ctx.Done():
				if ctx.Err() != nil {
					fmt.Printf("程序出现错误: %q", ctx.Err())
				} else {
					fmt.Println("程序被终止")
				}
				return
			default:
				readString, err := reader.ReadString('\n')
				if err != nil || err == io.EOF {
					break
				}
				fmt.Print(readString)
			}
		}
	}()
	return c.Run()
}

/*
func main() {
	cmd := exec.Command("/bin/bash", "-c", "")
	//创建获取命令输出管道
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Printf("Error:can not obtain stdout pipe for command:%s\n", err)
		return
	}

	//执行命令
	if err := cmd.Start(); err != nil {
		fmt.Println("Error:The command is err,", err)
		return
	}
	/*
	   	//读取所有输出
	   	bytes, err := ioutil.ReadAll(stdout)
	   	if err != nil {
	   		fmt.Println("ReadAll Stdout:", err.Error())
	   		return
	   	}

	   	if err := cmd.Wait(); err != nil {
	   		fmt.Println("wait:", err.Error())
	   		return
	   	}
	   	fmt.Printf("stdout:\n\n %s", bytes)
	   }
*/
/*
	outputBuf := bufio.NewReader(stdout)

	for {

		//一次获取一行,_ 获取当前行是否被读完
		output, _, err := outputBuf.ReadLine()
		if err != nil {

			// 判断是否到文件的结尾了否则出错
			if err.Error() != "EOF" {
				fmt.Printf("Error :%s\n", err)
			}
			return
		}
		fmt.Printf("%s\n", string(output))
	}

	//wait 方法会一直阻塞到其所属的命令完全运行结束为止
	if err := cmd.Wait(); err != nil {
		fmt.Println("wait:", err.Error())
		return
	}
}

/*
func Master() {
	for {
		fmt.Println("cccccccccccc --->  aaaaaaaa")
		time.Sleep(2 * time.Second)
	}
}

func main() {
	sigfile := "./myTest.pid"

	_, err := os.Stat(sigfile)
	if err == nil {
		fmt.Println("PID file exist.running...")
		os.Exit(0)
	}
	pidFileHandle, err := os.OpenFile(sigfile, os.O_RDONLY|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	go Master()
	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	for {
		<-c
		fmt.Println("wwwwwwwwwwww")
		err = pidFileHandle.Close()
		if err != nil {
			fmt.Println(err)
		}
		err = os.Remove(sigfile)
		if err != nil {
			fmt.Println(err)
		}
		time.Sleep(1 * time.Second)
		os.Exit(1)
	}
}

//1000瓶酒中有1瓶毒酒，10只老鼠，7天后毒性才发作，第8天要卖了，怎么求那瓶毒酒？
/*
func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	shift := 0
	row := make([]int, len(num2)+len(num1)+1)
	for i := len(num1) - 1; i >= 0; i-- {
		skew := 0
		for j := len(num2) - 1; j >= 0; j-- {
			tmp := stringToInteger(string(num1[i])) * stringToInteger(string(num2[j]))
			row[len(row)-skew-shift-1] += tmp
			skew++
		}
		shift++
	}
	row = SingleDigitsSlice(row)
	str := ""
	for _, v := range row {
		if str == "" && v == 0 {
			continue
		}
		str += integerToString(v)
	}
	return str
}
func SingleDigitsSlice(num []int) []int {
	for i := len(num) - 1; i >= 0; i-- {
		if num[i]/10 > 0 {
			num[i-1] += num[i] / 10
			num[i] %= 10
		}
	}
	return num
}
func stringToInteger(s string) int { //string转int
	m, _ := strconv.Atoi(s)
	return m
}
func integerToString(m int) string { //int转string
	s := strconv.Itoa(m)
	return s
}

func main() {
	s := "1239239293929"
	p := "4732847327428983428"
	fmt.Println(multiply(p, s))
}

/*
func maxArea(height []int) int {
	area := 0
	p := 0
	q := len(height) - 1
	for p < q {
		hl := MinNum(height[q], height[p])
		length := q - p
		if area < hl*length {
			area = hl * length
		}
		if height[q] > height[p] {
			p++
		} else {
			q--
		}
	}
	return area
}
func MaxNum(m, n int) int {
	if m > n {
		return m
	}
	return n
}
func MinNum(m, n int) int {
	if m < n {
		return m
	}
	return n
}
func main() {
	var s []int = []int{1, 8, 6, 2, 5, 4, 8, 3, 7}
	fmt.Println(maxArea(s))
}

/*
func candy(ratings []int) (ans int) {
	n := len(ratings)
	left := make([]int, n)
	for i, r := range ratings {
		if i > 0 && r > ratings[i-1] {
			left[i] = left[i-1] + 1
		} else {
			left[i] = 1
		}
	}
	fmt.Println(left)
	right := 0
	for i := n - 1; i >= 0; i-- {
		if i < n-1 && ratings[i] > ratings[i+1] {
			right++
		} else {
			right = 1
		}
		fmt.Println(right)
		ans += max(left[i], right)
	}
	return
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func main() {
	var s []int = []int{1, 1, 1, 2, 4, 3, 2, 1, 2, 2, 1}
	fmt.Println(candy(s))
}

/*
func candy(ratings []int) int {
	if len(ratings) == 0 {
		return 0
	}
	if len(ratings) == 1 {
		return 1
	}
	num := 0
	tmp := 0
	if ratings[1] > ratings[0] {
		num = 3
		tmp = 2
	} else if ratings[1] > ratings[0] {
		num = 3
		tmp = 1
	} else {
		num = 2
		tmp = 1
	}
	if len(ratings) == 2 {
		return num
	}
	for i := 2; i < len(ratings); i++ {
		if ratings[i] > ratings[i-1] {
			tmp += 1
			num += tmp
		} else if ratings[i] < ratings[i-1] {
			if len(ratings) > i+2 {

			}
		}
	}
}

/*
var factors = []int{2, 3, 5}

type hp struct{ sort.IntSlice }

func (h *hp) Push(v interface{}) { h.IntSlice = append(h.IntSlice, v.(int)) }
func (h *hp) Pop() interface{} {
	a := h.IntSlice
	v := a[len(a)-1]
	h.IntSlice = a[:len(a)-1]
	return v
}

func nthUglyNumber(n int) int {
	h := &hp{sort.IntSlice{1}}
	seen := map[int]struct{}{1: {}}
	for i := 1; ; i++ {
		x := heap.Pop(h).(int)
		if i == n {
			return x
		}
		for _, f := range factors {
			next := x * f
			if _, has := seen[next]; !has {
				heap.Push(h, next)
				seen[next] = struct{}{}
			}
		}
	}
}

/*
func nthUglyNumber(n int) int {
	if n == 1 {
		return 1
	}
	num := 1
	n = n - 1
	for n > 0 {
		num++
		if num%2 == 0 || num%5 == 0 || num%3 == 0 {
			n--
		}
	}
	return num
}
func chooseNumber(n int) bool {

}
func main() {
	fmt.Println(nthUglyNumber(12))
}

/*
type ListNode struct {
	Val  int
	Next *ListNode
}

func deleteDuplicates(cur *ListNode) *ListNode {
	if cur == nil {
		return cur
	}
	head := cur
	for head.Next != nil {
		if head.Val == head.Next.Val {
			head.Next = head.Next.Next
		} else {
			head = head.Next
		}
	}
	return cur
}
func main() {
	var lone *ListNode
	lone = &ListNode{Val: 1}
	l1 := lone
	l1.Next = &ListNode{Val: 3}
	l1 = l1.Next
	l1.Next = &ListNode{Val: 3}
	l1 = l1.Next
	fmt.Println(l1)
	fmt.Println(deleteDuplicates(l1).Val)
}

/*
func climbStairs(n int) int {
    p, q, r := 0, 0, 1
    for i := 1; i <= n; i++ {
        p = q
        q = r
        r = p + q
    }
    return r
}


/*
func plusOne(digits []int) []int {
	num := len(digits)
	digits[num-1] = digits[num-1] + 1
	for i := num - 1; i >= 0; i-- {
		if digits[i]/10 == 1 {
			if i != 0 {
				digits[i-1] += 1
				digits[i] = 0
			} else {
				digits[0] = 0
				digits = append([]int{1}, digits...)
			}
		}
	}
	return digits
}
func main() {
	num := []int{1, 0, 0, 0, 0}
	fmt.Println(plusOne(num))
}

/*
func lengthOfLastWord(s string) int {
	tmp := 0
	flag := false
	for i := len(s) - 1; i >= 0; i-- {
		if string(s[i]) != " " {
			flag = true
		} else {
			flag = false
			if tmp > 0 {
				break
			}
		}
		if flag {
			tmp++
		}
	}
	return tmp
}
func main() {
	fmt.Println(lengthOfLastWord("Hello World"))
	fmt.Println(lengthOfLastWord("   fly me   to   the moon  "))
}

/*
func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	return max
}

/*
func searchInsert(nums []int, target int) int {
	n := len(nums)
	left, right := 0, n-1
	ans := n
	for left <= right {
		mid := (right-left)>>1 + left
		if target <= nums[mid] {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return ans
}

/*
func strStr(haystack string, needle string) int {
	if haystack == needle {
		return 0
	}
	num := len(needle)
	for i := 0; i < len(haystack)-num+1; i++ {
		if needle == haystack[i:i+num] {
			return i
		}
	}
	return -1
}
func main() {
	s := "hello world"
	p := "ld"
	fmt.Println(strStr(s, p))
}

/*
func removeDuplicates(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	slow := 1
	for fast := 1; fast < n; fast++ {
		if nums[fast] != nums[fast-1] {
			nums[slow] = nums[fast]
			slow++
		}
	}
	fmt.Println(nums)
	return slow
}
func main() {
	var arr []int = []int{1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6}
	fmt.Println(removeDuplicates(arr))
}

/*
func removeDuplicates(nums []int) int {
	length := len(nums)
	p1 := 0
	p2 := 1
	for p2 < len(nums) {
		if nums[p1] == nums[p2] {
			length--
			p2++
		} else {
			p1 = p2
			p2++
		}
	}
	return length
}
func main() {
	var arr []int = []int{1, 1, 2, 3, 4, 4, 5, 5, 6, 6}
	fmt.Println(removeDuplicates(arr))
}

/*
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	prehead := &ListNode{0, nil}
	prev := prehead
	if l1 == nil {
		prev.Next = l2
	} else {
		prev.Next = l1
	}
	for {
		if l1 == nil {
			break
		}
		if l2 == nil {
			break
		}
		if l1.Val <= l2.Val {
			prev.Next = l1
			l1 = l1.Next
		} else {
			prev.Next = l2
			l2 = l2.Next
		}
		prev = prev.Next
	}
	return prehead.Next
}
func main() {
	var lone *ListNode
	lone = &ListNode{Val: 1}
	l1 := lone
	l1.Next = &ListNode{Val: 2}
	l1 = l1.Next
	l1.Next = &ListNode{Val: 3}
	l1 = l1.Next
	var ltwo *ListNode
	ltwo = &ListNode{Val: 1}
	l2 := ltwo
	l2.Next = &ListNode{Val: 2}
	l2 = l2.Next
	l2.Next = &ListNode{Val: 4}
	l2 = l2.Next
	fmt.Println(mergeTwoLists(lone, ltwo).Next.Next.Next.Val)
	//fmt.Println(lone.Next.Next.Val)
	//fmt.Println(ltwo)
}

/*
func isValid(s string) bool {
	n := len(s)
	if n%2 == 1 {
		return false
	}
	pairs := map[byte]byte{
		')': '(',
		']': '[',
		'}': '{',
	}
	stack := []byte{}
	for i := 0; i < n; i++ {
		if pairs[s[i]] > 0 {
			if len(stack) == 0 || stack[len(stack)-1] != pairs[s[i]] {
				return false
			}
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, s[i])
		}
	}
	return len(stack) == 0
}
func main() {
	fmt.Println(isValid("([)]"))
	fmt.Println(isValid("()[]{[]}"))
	fmt.Println(isValid("({[()]})"))
	fmt.Println(isValid("()[]{}{"))
}

/*
func isPalindrome(x int) bool {
	sum := x
	if x == 0 {
		return true
	}
	if x < 0 || x%10 == 0 {
		return false
	}
	tmp := 0
	for x > 0 {
		tmp = tmp * 10
		tmp += x % 10
		x /= 10
	}
	fmt.Println(tmp)
	if tmp != sum {
		return false
	}
	return true
}
func main() {
	fmt.Println(isPalindrome(123))
	fmt.Println(isPalindrome(121))
	fmt.Println(isPalindrome(10))
}

/*
func romanToInt(s string) int {
	datamap := map[string]int{
		"I":  1,
		"V":  5,
		"X":  10,
		"L":  50,
		"C":  100,
		"D":  500,
		"M":  1000,
		"IX": 9,
		"IV": 4,
		"XL": 40,
		"XC": 90,
		"CD": 400,
		"CM": 900,
	}
	num := 0
	for i := 0; i < len(s); i++ {
		if len(s) == 1 {
			num = datamap[s]
			return num
		}
		if len(s) > i+1 && datamap[string(s[i])] < datamap[string(s[i+1])] {
			num += datamap[string(s[i:i+2])]
			i++
		} else {
			num += datamap[string(s[i])]
		}
	}
	return num
}
func main() {
	var s string = "D"
	fmt.Println(romanToInt(s))
}

/*
func longestCommonPrefix(strs []string) string {
	tmp := ""
	if len(strs) == 0 {
		return tmp
	}
	for i := 0; i < len(strs[0]); i++ {
		datamap := make(map[string]bool)
		datamap[string(strs[0][i])] = false
		for j := 1; j < len(strs); j++ {
			if len(strs[j]) == 0 {
				return tmp
			}
			if len(strs[j]) < i+1 {
				return tmp
			}
			if _, ok := datamap[string(strs[j][i])]; !ok {
				return tmp
			}
		}
		tmp += string(strs[0][i])
	}
	return tmp
}
func main() {
	var s []string = []string{"rdog", "racecar", "racar"}
	fmt.Println(longestCommonPrefix(s))
}

/*
import (
	"fmt"
	"time"
)

func Process2(tasks []string) {
	for _, task := range tasks {
		// 启动协程并发处理任务
		go func(t string) {
			fmt.Printf("Worker start process task: %s\n", t)
		}(task)
	}
}
func Process1(tasks []string) {
	for _, task := range tasks {
		// 启动协程并发处理任务
		go func() {
			fmt.Println(task)
		}()
	}
}
func main() {
	s := []string{"a", "b", "c"}
	Process1(s)
	Process2(s)
	time.Sleep(2 * time.Second)
}

/*
func main() {
	for i := 0; i < 100; i++ {
		url := "http://monitor.static01.ucloud.com.cn/ucloud.123123"
		_, err := GetRequestTool(url, "11-22-333")
		if err != nil {
			fmt.Println(err)
		}
	}

}
func IntegerToString(m int) string { //int转string
	s := strconv.Itoa(m)
	return s
}
func GetRequestTool(Url string, uuid string) ([]byte, error) {
	Resp, errHttp := http.Get(Url)
	if errHttp != nil {
		return nil, errHttp
	}
	defer Resp.Body.Close()
	if Resp.StatusCode != 200 {
		return nil, errors.New("code ! 200  code is : " + IntegerToString(Resp.StatusCode))
	}
	body, err := ioutil.ReadAll(Resp.Body)
	if err != nil {
		return nil, err
	}
	return body, nil
}

/*
// 通过递归反转单链表
type Node struct {
	Value    int
	NextNode *Node
}

func Param(node *Node) {
	for node != nil {
		fmt.Print(node.Value, "--->")
		node = node.NextNode
	}
	fmt.Println()
}

func reverse(headNode *Node) *Node {
	if headNode == nil {
		return headNode
	}
	if headNode.NextNode == nil {
		return headNode
	}
	fmt.Println(headNode.Value)
	fmt.Println(headNode.NextNode.Value)
	var newNode = reverse(headNode.NextNode)
	headNode.NextNode.NextNode = headNode
	headNode.NextNode = nil
	return newNode
}
func main() {
	var node1 = &Node{}
	node1.Value = 1
	node2 := new(Node)
	node2.Value = 2
	node3 := new(Node)
	node3.Value = 3
	node4 := new(Node)
	node4.Value = 4
	node5 := new(Node)
	node5.Value = 5
	node6 := new(Node)
	node6.Value = 6
	node7 := new(Node)
	node7.Value = 7
	node1.NextNode = node2
	node2.NextNode = node3
	node3.NextNode = node4
	node4.NextNode = node5
	node5.NextNode = node6
	node6.NextNode = node7
	Param(node1)
	reverseNode := reverse(node1)
	Param(reverseNode)
}
func IntegerToString(m int) string { //int转string
	s := strconv.Itoa(m)
	return s
}

/*
func main() {
	var ProSlice []string = []string{"默认", "北京市", "天津市", "上海市", "重庆市", "云南省", "内蒙古", "吉林省", "四川省", "宁夏", "安徽省", "山东省", "山西省", "广东省", "广西", "新疆", "江苏省", "江西省", "河北省", "河南省", "浙江省", "浙江省", "海南省", "湖北省", "湖南省", "甘肃省", "福建省", "西藏", "贵州省", "辽宁省", "陕西省", "青海省", "黑龙江省", "台湾省", "香港", "澳门"}
	//var IspSlice []string = []string{"默认","电信","联通","教育网","移动","e家宽","铁通","世纪互联","艾普","长宽","方正宽带","华数网通","天威视讯","科技网","广电网","东方有线","歌华有线","IX","湖南有线","皓宽","东方网信","吉视传媒","阿里云","珠江数码"}
	s := "广东"
	for _, v := range ProSlice {
		if find := strings.Contains(v, s); find {
			fmt.Println(s)
			fmt.Println(v)
		}
	}
}




func main() {
	fmt.Println(StringToInteger("6507"))
}

/*
type Person struct {
	age int
}

func (p Person) Elegance() int {
	return p.age
}

func (p *Person) GetAge() int {
	p.age += 1
	return p.age
}

func main() {
	// p1 是值类型
	p := Person{age: 18}

	// 值类型 调用接收者也是值类型的方法
	fmt.Println(p.Elegance())

	// 值类型 调用接收者是指针类型的方法
	fmt.Println(p.GetAge())

	// ----------------------

	// p2 是指针类型
	p2 := &Person{age: 100}

	// 指针类型 调用接收者是值类型的方法
	fmt.Println(p2.Elegance())

	// 指针类型 调用接收者也是指针类型的方法
	fmt.Println(p2.GetAge())
}

/*
type M struct {
	GlbVer int
	S      string
}

func main() {
	v := 20
	rcdData := make([]*M, 0)
	m := &M{}
	m.GlbVer = 10
	rcdData = append(rcdData, m)
	m = &M{}
	m.GlbVer = 18
	rcdData = append(rcdData, m)
	m = &M{}
	m.GlbVer = 20
	rcdData = append(rcdData, m)
	m = &M{}
	m.GlbVer = 22
	rcdData = append(rcdData, m)
	m = &M{}
	m.GlbVer = 17
	rcdData = append(rcdData, m)
	for i := 0; i < len(rcdData); {
		if rcdData[i].GlbVer <= v {
			tmp := rcdData[i].GlbVer
			rcdData = append(rcdData[:i], rcdData[i+1:]...)
			DeleteFlag := true
			for j := 0; j < len(rcdData); j++ {
				if rcdData[j].GlbVer == tmp {
					DeleteFlag = false
				}
			}
			if DeleteFlag {
				fmt.Println("tmtmtmtmtmtm")
				fmt.Println(tmp)
			}
		} else {
			i++
		}
	}
	fmt.Println(rcdData[0])
	fmt.Println(len(rcdData))
}

/*
var (
	counter int32          //计数器
	wg      sync.WaitGroup //信号量
)

func main() {
	sNum := 0
	threadNum := 10000
	wg.Add(threadNum)
	for i := 0; i < threadNum; i++ {
		go incCounter(i, &sNum)
	}
	wg.Wait()
}

func incCounter(index int, sNum *int) {
	defer wg.Done()

	spinNum := 0
	for {
		old := counter
		ok := atomic.CompareAndSwapInt32(&counter, old, old+1)
		if ok {
			break
		} else {
			spinNum++
		}
	}
	*sNum++
	fmt.Printf("thread,%d,spinnum,%d, sNum,%d\n", index, spinNum, *sNum)
}

/*
func main() {
	fmt.Println(increase(1))
}
func increase(d int) (ret int) {
	fmt.Println(ret)
	defer func() {
		fmt.Println(ret)
		ret++
		fmt.Println(ret)
	}()
	return d
}

/*
func main() {
	defer func() {
		fmt.Println("Try to recover the panic")
		if p := recover(); p != nil {
			fmt.Println("recover the panic : ", p)
		}
	}()
	var mutex sync.Mutex
	fmt.Println("begin lock")
	mutex.Lock()
	fmt.Println("get locked")
	fmt.Println("unlock lock")
	mutex.Unlock()
	fmt.Println("lock is unlocked")
	fmt.Println("unlock lock again")
	mutex.Unlock()
}

/*
func test() {
	for {
		time.Sleep(1 * time.Second)
		fmt.Println("aaaaaaaa")
		time.Sleep(1 * time.Second)
		timeStr := time.Now().Format("2006-01-02 15:04:05") //当前时间的字符串，2006-01-02 15:04:05据说是golang的诞生时间，固定写法
		fmt.Println(timeStr)
		fmt.Println("bbbbbbbb")
	}
}
func main() {
	var wg sync.WaitGroup
	wg.Add(1)
	go test()

	wg.Wait()
}

/*
func main() {
	DM5 := fmt.Sprintf("%x", md5.Sum([]byte("txmov2.a.yximgs.comtxmov2.a.yximgs.comtxmov2.a.yximgs.com")))
	fmt.Println(DM5)
}

/*
func main() {
	var arr []string = []string{"txmov2.a.yximgs.com", "txmov2.a.yximgs.com", "txmov2.a.yximgs.com"}
	hmd := md5.New()
	for _, v := range arr {
		io.WriteString(hmd, v)
	}
	DM5 := fmt.Sprintf("%x", hmd.Sum(nil))
	fmt.Println(DM5)
}

/*
type People interface {
	Kid() bool
	Man(bool) bool
	Un() bool
}
type peopler struct{}

func (*peopler) Man(flag bool) bool {
	if !flag {
		fmt.Println("Man")
	}
	return true
}
func (*peopler) Kid() bool {
	fmt.Println("Kid")
	return true
}
func (p *peopler) Un() bool {
	f := p.Kid()
	if f {
		fmt.Println("Un")
	}
	return true
}
func NewPeople() People {
	return &peopler{}
}
func AllPeople() {
	people := NewPeople()
	flag := people.Kid()
	_ = people.Man(flag)
	_ = people.Un()
}
func main() {
	AllPeople()
}

/*
func fibonacci(ch chan int, done chan struct{}) {
	x, y := 0, 1
	for {
		select {
		case ch <- x:
			x, y = y, x+y
		case <-done:
			fmt.Println("over")
			return
		}
	}
}
func main() {
	ch := make(chan int)
	done := make(chan struct{})
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-ch)
		}
		done <- struct{}{}
	}()
	fibonacci(ch, done)
}

/*
func main() {
	s := `{"ll":4, "qq":"kkk"}`
	est := json.Valid([]byte(s))
	fmt.Println(est)
	s1 := `{"\'ll":4, "qq":"kkk"]}`
	est = json.Valid([]byte(s1))
	fmt.Println(est)
}

/*
func main() {
	new := `server {
	listen 443 ssl;
	listen [::]:443 ssl;
	server_name note.dmgc.us;

	ssl_certificate certs/note.dmgc.us.crt;
	ssl_certificate_key certs/note.dmgc.us.key;

	ssl_dhparam /usr/local/nginx/conf/dhparam.pem;
	ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
	ssl_prefer_server_ciphers  on;

	set $real_uri $uri;
	location / {
		more_set_headers "Server: $upstream_http_server";
		proxy_set_header  X-Real-IP  $remote_addr;
		proxy_set_header Host "note.dmgc.us";
		proxy_set_header X-Protocol https;
		proxy_http_version 1.1;
		proxy_buffering off;
		proxy_pass http://http_backend;
	}

}`
	new = strings.Replace(new, " ", "", -1)
	new = strings.Replace(new, "\n", "", -1)
	new = strings.Replace(new, "\t", "", -1)
	new = strings.Replace(new, "\r", "", -1)
	fmt.Println(len(new))
	tmp := make([]int, 0)
	for i := 0; i < len(new)-7; i++ {
		if new[i:i+7] == "server{" {
			tmp = append(tmp, i)
		}
	}
	fmt.Println(tmp)
}

/*
func main() {
	s := "server {listen       443 ssl;listen  [::]:443 ssl;server_name  m2.mogucdn.com;ssl_certificate      certs/m2.mogucdn.com.crt;ssl_certificate_key  certs/m2.mogucdn.com.key;ssl_dhparam /usr/local/nginx/conf/dhparam.pem;ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;ssl_ciphers  'TLS13-AES-256-GCM-SHA384:TLS13-CHACHA20-POLY1305-SHA256:TLS13-AES-128-GCM-SHA256:TLS13-AES-128-CCM-8-SHA256:TLS13-AES-128-CCM-SHA256:EECDH+AESGCM:EDH+AESGCM:AES128+EECDH:AES128+EDH:HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4';ssl_prefer_server_ciphers  on;set $real_uri $uri;location / {"
	s = strings.Replace(s, " ", "", -1)
	s = strings.Replace(s, "\n", "", -1)
	s = strings.Replace(s, "\t", "", -1)
	s = strings.Replace(s, "\r", "", -1)
	fmt.Println(len(s))
}


/*
func main() {
	a := ""
	already := make([]string, 0)
	already = strings.Split(a, ",")
	fmt.Println(len(already))
	fmt.Println(already[0])
}

/*
func main() {
	rand.Seed(time.Now().Unix())
	a := rand.Intn(86400 * 7)
	fmt.Println(a)
}

/*
func main() {
	go A()
	for {
		time.Sleep(time.Hour)
	}
}
func A() {
	i := 0
	for {
		i++
		if i == 1000000 {
			fmt.Println(i)
			os.Exit(0)
			return
		}
	}
}

/*
func IsDomain(domain string) bool {
	if len(domain) <= 2 {
		return false
	}
	if string(domain[0]) == "." {
		domain = domain[1:]
	}
	pattern := `^[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+\.?$`
	matched, err := regexp.MatchString(pattern, domain)
	if err != nil {
		return false
	}
	if matched {
		return true
	}
	return false
}
func CheckIsDomain(Domain string) bool {
	Rescode := false
	flag := false
	if len(Domain) < 3 {
		return false
	}
	address := net.ParseIP(Domain)
	if address != nil {
		return false
	}
	StrArr := strings.Split(Domain, ".")
	if len(StrArr) < 2 {
		return false
	}
	for i := 0; i < len(StrArr); i++ {
		flag, _ = regexp.MatchString("^[A-Za-z0-9]+$", StrArr[i])
		if flag == false {
			for j := 0; j < len(StrArr[i]); j++ {
				if string(StrArr[i][j]) != "-" && string(StrArr[i][j]) != "_" {
					sign, _ := regexp.MatchString("^[A-Za-z0-9]+$", string(StrArr[i][j]))
					if sign == false {
						Rescode = false
						break
					} else {
						Rescode = true
					}
				} else {
					if j == len(StrArr[i]) {
						Rescode = true
					}
				}
			}
		} else {
			Rescode = true
		}
		if Rescode == false {
			break
		}
	}
	return Rescode
}
func main() {
	fmt.Println(IsDomain("1.1.1.1"))
	fmt.Println(CheckIsDomain("1.1.1.1"))
}

/*
func main() {
	f, err := os.OpenFile("/Users/user/Desktop/Development/myTest/DomainConf.txt", os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0766)
	if err != nil {

		return
	}
	f.WriteString("dasjjda111")
	f.Close()
	fmt.Println(reflect.TypeOf(f))
}

/*
func main() {
	a := flag.String("lll", "", "bao")
	b := flag.String("kkk", "liuliu", "bei")
	c := flag.Bool("ppp", false, "pig")
	flag.Parse()
	x := *a
	y := *b
	z := *c
	fmt.Println(*b)
	if len(x) > 0 {
		fmt.Println("12345567")
	}
	if len(y) > 0 {
		fmt.Println("asdfghjkl")
	}
	if !z {
		fmt.Println(z)
	}
}

/*
var cond sync.Cond // 创建全局条件变量

// 生产者
func producer(out chan<- int, idx int) {
	for {
		cond.L.Lock()       // 条件变量对应互斥锁加锁
		for len(out) == 3 { // 产品区满 等待消费者消费
			cond.Wait() // 挂起当前协程， 等待条件变量满足，被消费者唤醒
		}
		num := rand.Intn(1000) // 产生一个随机数
		out <- num             // 写入到 channel 中 （生产）
		fmt.Printf("%dth 生产者，产生数据 %3d, 公共区剩余%d个数据\n", idx, num, len(out))
		cond.L.Unlock()         // 生产结束，解锁互斥锁
		cond.Signal()           // 唤醒 阻塞的 消费者
		time.Sleep(time.Second) // 生产完休息一会，给其他协程执行机会
	}
}

//消费者
func consumer(in <-chan int, idx int) {
	for {
		cond.L.Lock()      // 条件变量对应互斥锁加锁（与生产者是同一个）
		for len(in) == 0 { // 产品区为空 等待生产者生产
			cond.Wait() // 挂起当前协程， 等待条件变量满足，被生产者唤醒
		}
		num := <-in // 将 channel 中的数据读走 （消费）
		fmt.Printf("---- %dth 消费者, 消费数据 %3d,公共区剩余%d个数据\n", idx, num, len(in))
		cond.L.Unlock()                    // 消费结束，解锁互斥锁
		cond.Signal()                      // 唤醒 阻塞的 生产者
		time.Sleep(time.Millisecond * 500) //消费完 休息一会，给其他协程执行机会
	}
}
func main() {
	rand.Seed(time.Now().UnixNano()) // 设置随机数种子
	quit := make(chan bool)          // 创建用于结束通信的 channel

	product := make(chan int, 3) // 产品区（公共区）使用channel 模拟
	cond.L = new(sync.Mutex)     // 创建互斥锁和条件变量

	for i := 0; i < 5; i++ { // 5个消费者
		go producer(product, i+1)
	}
	for i := 0; i < 3; i++ { // 3个生产者
		go consumer(product, i+1)
	}
	<-quit // 主协程阻塞 不结束
}

/*

func producer(out chan<- int) {
	for {
		if len(out) == 0 {
			num := rand.Intn(1000)
			out <- num
			fmt.Printf("1111111111111111\n")
			fmt.Println(len(out))
			time.Sleep(time.Millisecond * 200)
		}
	}
}
func consumer_1(in <-chan int) {
	for {
		if len(in) > 0 {
			<-in
			fmt.Printf("pppppppppppppppppppppppppppp\n")
			fmt.Println(len(in))
			time.Sleep(time.Second * 1)
		}
	}
}
func consumer_2(in <-chan int) {
	for {
		if len(in) > 0 {
			<-in
			fmt.Printf("ssssssssssssssss\n")
			fmt.Println(len(in))
			time.Sleep(time.Second * 3)
		}
	}
}
func main() {
	quit := make(chan int)
	product := make(chan int, 1)
	go producer(product)
	go consumer_1(product)
	go consumer_2(product)
	<-quit
}

/*
func main() {
	ch := make(chan int, 3)
	ch <- 1
	ch <- 2
	ch <- 3
	fmt.Println(<-ch)
	fmt.Println(len(ch))

}

/*
func A(lock *sync.Mutex, count *int) {
	for {
		lock.Lock()
		*count++
		fmt.Println("212212121121")
		fmt.Println(*count)
		lock.Unlock()
		time.Sleep(time.Second * 10)
	}
}
func B(lock *sync.Mutex, count *int) {
	for {
		lock.Lock()
		*count++
		fmt.Println("sssssssssssss")
		fmt.Println(*count)
		lock.Unlock()
		time.Sleep(time.Second * 1)
	}

}
func main() {
	lock := &sync.Mutex{}
	var count int = 0
	go A(lock, &count)
	go B(lock, &count)
	for {
		time.Sleep(time.Hour * 1)
	}
}

/*
/*
var cond sync.Cond // 创建全局条件变量
func main() {
	cond.L = new(sync.Mutex)
	ch := make(chan int)
	n := a(ch, cond.L)
	fmt.Println((n))
}
func a(ch <-chan int, cond.L sync.Locker) int {
	return 1
}

/*
func main() {
	var i int = 0
	for {
		i++
		fmt.Println(i)
		if i%2 == 0 {
			goto Sleep
		}
		fmt.Println(i)
	Sleep:
		time.Sleep(time.Second * 2)
	}
}


/*
func main() {
	var s []string = []string{"dasjdjas", "dsajdaskdlka", "Dsadjhaskjdkas", "Diif"}
	lt := ""
	h := md5.New()
	for _, v := range s {
		io.WriteString(h, v)
		lt += v
	}
	m5 := fmt.Sprintf("%x", h.Sum(nil))
	m6 := fmt.Sprintf("%x", md5.Sum([]byte(lt)))
	fmt.Println(m5)
	fmt.Println(m6)
}

/*
func main() {
	s := ``
	p := ``
	s = strings.Replace(s, " ", "", -1)
	s = strings.Replace(s, "\n", "", -1)
	s = strings.Replace(s, "\t", "", -1)
	s = strings.Replace(s, "\r", "", -1)
	p = strings.Replace(p, " ", "", -1)
	p = strings.Replace(p, "\n", "", -1)
	p = strings.Replace(p, "\t", "", -1)
	p = strings.Replace(p, "\r", "", -1)
	fmt.Println(len(s))
	fmt.Println(len(p))
	fmt.Println(s)
	fmt.Println(p)
}

/*
var w sync.WaitGroup

func Coroutine() {
	w.Add(1000)
	for i := 0; i < 1000; i++ {
		num := rand.Intn(9)
		go func(n int) {
			time.Sleep(time.Duration(num+1) * time.Second)
			A(n)
			w.Done()
		}(i)
	}
	w.Wait()
}
func A(a int) {
	m := make(map[int]int)
	for j := 0; j < 5000; j++ {
		m[j] = j
	}
	fmt.Println("ip" + IntegerToString(a) + " : " + IntegerToString(len(m)))
}
func IntegerToString(m int) string { //int转string
	s := strconv.Itoa(m)
	return s
}
func main() {
	Coroutine()
}

/*
func getValueByKeyFromConf(key, origin_s string) string {
	idx1 := strings.Index(origin_s, key)
	if idx1 == -1 {
		return ""
	}
	tmp_conf := origin_s[idx1 : len(origin_s)-idx1]
	idx2 := strings.Index(tmp_conf, "\n")
	if idx2 == -1 {
		return ""
	}
	r_str := origin_s[idx1:idx2]
	v_c := strings.Split(r_str, "=")
	if len(v_c) == 2 {
		return DeletePreAndSufSpace(v_c[1])
	}
	return ""
}
func DeletePreAndSufSpace(str string) string {
	strList := []byte(str)
	spaceCount, count := 0, len(strList)
	for i := 0; i <= len(strList)-1; i++ {
		if strList[i] == 32 {
			spaceCount++
		} else {
			break
		}
	}

	strList = strList[spaceCount:]
	spaceCount, count = 0, len(strList)
	for i := count - 1; i >= 0; i-- {
		if strList[i] == 32 {
			spaceCount++
		} else {
			break
		}
	}

	return string(strList[:count-spaceCount])
}
func main() {
	fmt.Println(getValueByKeyFromConf("origin_with_host", "origin_with_host   =  dasjdjasj   \n dasjdjasj =origin_with_host"))
}

/*
func CloseHttpsNginxOperation(xmlL1Nginx string) (NginxConfL1 string) {
	if find := strings.Contains(xmlL1Nginx, "ssl_protocols"); !find {
		NginxConfL1 = xmlL1Nginx
		return
	}
	nginx_arr := make([]string, 0)
	for {
		server_first := strings.Index(xmlL1Nginx, "server {")
		if server_first < 0 {
			break
		}
		server_second := strings.Index(xmlL1Nginx[server_first+1:], "server {")
		if server_second < 0 {
			nginx_arr = append(nginx_arr, xmlL1Nginx[server_first:])
			xmlL1Nginx = ""
		} else {
			nginx_arr = append(nginx_arr, xmlL1Nginx[server_first:server_second+1])
			xmlL1Nginx = xmlL1Nginx[server_second+1:]
		}
	}
	for i := 0; i < len(nginx_arr); i++ {
		if find := strings.Contains(nginx_arr[i], "ssl_protocols"); find {
			nginx_arr = append(nginx_arr[:i], nginx_arr[i+1:]...)
		}
	}
	for _, v := range nginx_arr {
		NginxConfL1 += v + "\n"
	}
	return
}
func main() {
	s := `server {
		listen       443 ssl;
		listen  [::]:443 ssl;
		server_name  {domain_template};

		ssl_certificate      certs/{domain_template}.crt;
		ssl_certificate_key  certs/{domain_template}.key;

		ssl_dhparam /usr/local/nginx/conf/dhparam.pem;
		ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
		ssl_ciphers  'TLS13-AES-256-GCM-SHA384:TLS13-CHACHA20-POLY1305-SHA256:TLS13-AES-128-GCM-SHA256:TLS13-AES-128-CCM-8-SHA256:TLS13-AES-128-CCM-SHA256:EECDH+AESGCM:EDH+AESGCM:AES128+EECDH:AES128+EDH:HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4';
		ssl_prefer_server_ciphers  on;

		set $real_uri $uri;
		location / {
			more_set_headers "Server: $upstream_http_server";
			proxy_set_header  X-Real-IP  $remote_addr;
			proxy_set_header  X-Forwarded-For  $remote_addr;
			proxy_set_header Host "{domain_template}";
			proxy_set_header X-Protocol https;
			proxy_http_version 1.1;
			proxy_buffering off;
			proxy_pass http://http_backend;
		}

	}
	server {
			listen       80;
			server_name  {domain_template};

			gzip  on;
			gzip_min_length 1k;
			gzip_buffers 4 16k;
			gzip_comp_level 2;
			gzip_types text/plain text/css text/javascript text/xml application/x-javascript application/json application/xml application/xml+rss application/javascript model/vnd.collada+xml;
			gzip_vary on;
			gzip_disable "MSIE [1-6]\.";
			gzip_http_version 1.0;
			location /nginx-vts-status-0528 {
					vhost_traffic_status_display;
					vhost_traffic_status_display_format json;
					allow 127.0.0.1;
					deny all;
			}

			location / {
					proxy_set_header X-Real-Port $remote_port;
					proxy_set_header X-Connection $connection;
					if ($request_method !~ "GET|HEAD|POST|PURGE|FETCH|QFETCH|OPTIONS") {
							return 403;
					}
					proxy_set_header  X-Real-IP  $remote_addr;
					proxy_set_header Host $host;
					proxy_buffering off;
					proxy_http_version 1.1;
					proxy_set_header Connection "";
					proxy_pass   http://http_backend;
			}
	}
	server {
		listen       443 ssl;
		listen  [::]:443 ssl;
		server_name  {domain_template};

		ssl_certificate      certs/{domain_template}.crt;
		ssl_certificate_key  certs/{domain_template}.key;

		ssl_dhparam /usr/local/nginx/conf/dhparam.pem;
		ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
		ssl_ciphers  'TLS13-AES-256-GCM-SHA384:TLS13-CHACHA20-POLY1305-SHA256:TLS13-AES-128-GCM-SHA256:TLS13-AES-128-CCM-8-SHA256:TLS13-AES-128-CCM-SHA256:EECDH+AESGCM:EDH+AESGCM:AES128+EECDH:AES128+EDH:HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4';
		ssl_prefer_server_ciphers  on;

		set $real_uri $uri;
		location / {
			more_set_headers "Server: $upstream_http_server";
			proxy_set_header  X-Real-IP  $remote_addr;
			proxy_set_header  X-Forwarded-For  $remote_addr;
			proxy_set_header Host "{domain_template}";
			proxy_set_header X-Protocol https;
			proxy_http_version 1.1;
			proxy_buffering off;
			proxy_pass http://http_backend;
		}

	}
	`
	fmt.Println(CloseHttpsNginxOperation(s))
}

/*
func main() {
	var path = "./looper"
	flag := true
	_, err := os.Stat(path)
	if err != nil {
		flag = false
	}
	if os.IsNotExist(err) {
		flag = false
	}
	if !flag {
		err = os.MkdirAll(path, os.ModePerm)
		if err != nil {
			fmt.Println(err)
			return
		}
	}

}

/*
func main() {
	err := os.Remove("./a.js")
	if err != nil {
		fmt.Println("dasjdkask")
		fmt.Println(err)
	}
}

/*
func main() {
	files, _ := ioutil.ReadDir("./")
	for _, f := range files {
		fmt.Println(f.Name())
	}
}

/*
func main() {
	f, err := os.OpenFile("./liujun.txt", os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		fmt.Println(err)
	}
	f.WriteString("doaosdoaso")
	f.Close()
}

/*
//  判断文件夹是否存在
func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
func main() {
	flag, err := PathExists("/Users/user/Desktop/Development/yinzhengjie/golang/code")
	if err != nil {
		fmt.Errorf("error", "error", err.Error())
	}
	fmt.Println(flag)
	if flag == false {
		os.MkdirAll("/Users/user/Desktop/Development/yinzhengjie/golang/code", os.ModePerm)
	}
}

/*  创建目录
var (
	OneLevelDirectory   = "./yinzhengjie"
	MultilevelDirectory = "/Users/user/Desktop/Development/yinzhengjie/golang/code"
)

func main() {
	//os.Mkdir(OneLevelDirectory, 0777)      //创建名称为OneLevelDirectory的目录，设置权限为0777。相当于Linux系统中的“mkdir yinzhengjie”
	os.MkdirAll(MultilevelDirectory, 0777) //创建MultilevelDirectory多级子目录，设置权限为0777。相当于Linux中的 “mkdir -p yinzhengjie/golang/code”
}
/* 删除目录
func main() {
    err := os.Remove(MultilevelDirectory) //删除名称为OneLevelDirectory的目录，当目录下有文件或者其他目录是会出错。
    if err != nil {
        fmt.Println(err)
    }
    os.RemoveAll(OneLevelDirectory) //根据path删除多级子目录，如果 path是单个名称，那么该目录不删除。
}
/*
func countPrimes(n int) int {
	count := 0
	signs := make([]bool, n)
	for i := 2; i < n; i++ {
		if signs[i] {
			continue
		}
		count++
		for j := 2 * i; j < n; j += i {
			fmt.Println(j + 100)
			fmt.Println(i + 1000)
			signs[j] = true
		}
	}
	return count
}
func main() {
	fmt.Println(countPrimes(10))
}

/*
type Subject struct {
	observers []Observer
	context   string
}

func NewSubject() *Subject {
	return &Subject{
		observers: make([]Observer, 0),
	}
}

func (s *Subject) Attach(o Observer) {
	s.observers = append(s.observers, o)
}

func (s *Subject) notify() {
	for _, o := range s.observers {
		o.Update(s)
	}
}

func (s *Subject) UpdateContext(context string) {
	s.context = context
	s.notify()
}

type Observer interface {
	Update(*Subject)
}

type Reader struct {
	name string
}

func NewReader(name string) *Reader {
	return &Reader{
		name: name,
	}
}

func (r *Reader) Update(s *Subject) {
	fmt.Printf("%s receive %s\n", r.name, s.context)
}
func ExampleObserver() {
	subject := NewSubject()
	reader1 := NewReader("reader1")
	reader2 := NewReader("reader2")
	reader3 := NewReader("reader3")
	subject.Attach(reader1)
	subject.Attach(reader2)
	subject.Attach(reader3)

	subject.UpdateContext("observer mode")
	// Output:
	// reader1 receive observer mode
	// reader2 receive observer mode
	// reader3 receive observer mode
}
func main() {
	ExampleObserver()
}

/*
type People interface {
	Kid() bool
	Man() bool
}
type peopler struct{}

func (*peopler) Man() bool {
	fmt.Println("Man")
	return true
}
func (*peopler) Kid() bool {
	fmt.Println("Kid")
	return true
}
func NewPeople() People {
	return &peopler{}
}
func AllPeople() {
	people := NewPeople()
	_ = people.Kid()
	_ = people.Man()
}
func main() {
	AllPeople()
}

/*
//Singleton 是单例模式类
type Singleton struct{}

var singleton *Singleton
var once sync.Once

//GetInstance 用于获取单例模式对象
func GetInstance() *Singleton {
	once.Do(func() {
		singleton = &Singleton{}
	})

	return singleton
}

const parCount = 100

func TestSingleton() {
	ins1 := GetInstance()
	ins2 := GetInstance()
	if ins1 != ins2 {
		fmt.Println("instance is not equal")
	}
}

func TestParallelSingleton() {
	wg := sync.WaitGroup{}
	wg.Add(parCount)
	instances := [parCount]*Singleton{}
	for i := 0; i < parCount; i++ {
		go func(index int) {
			instances[index] = GetInstance()
			fmt.Println(index)
			wg.Done()
		}(i)
	}
	wg.Wait()
	for i := 1; i < parCount; i++ {
		if instances[i] != instances[i-1] {
			fmt.Println("instance is not equal face")
		}
	}
}
func main() {
	TestParallelSingleton()
	TestSingleton()
}

/*
func isPerfectSquare(num int) bool {
	if num == 1 {
		return true
	}
	if num == 2 || num == 3 {
		return false
	}
	for i := 1; i < num/2; i++ {
		if i*i == num {
			return true
		}
	}
	return false
}
func main() {
	fmt.Println(isPerfectSquare(16))
}

/*
func main() {
	var num int = 30
	count := 1
	for num > 0 {
		count *= num
		num--
	}
	fmt.Println(IntegerToString(count))
}
func IntegerToString(m int) string { //int转string
	s := strconv.Itoa(m)
	return s
}

/*
func majorityElement(nums []int) int {
	t := nums[0]
	sum := 0
	for i := 0; i < len(nums); i++ {
		// 遍历
		// 如果与我们假设的值相等 就加一（投一票）
		// 如果与不相等 就减一票
		if nums[i] == t {
			sum++
		} else {
			sum--
		}

		// 如果票数<0了 就把前面的都放弃掉 从当前这个数假设为新的主要元素
		if sum < 0 {
			sum = 0
			t = nums[i]
		}
	}

	// 遍历计数
	cnt := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == t {
			cnt++
		}
	}

	// 超过一半返回该主要元素 否则返回-1
	if cnt > len(nums)/2 {
		return t
	} else {
		return -1
	}
}
func main() {
	fmt.Println(majorityElement([]int{3, 2, 1}))
}

/*
func twoSum(nums []int, target int) []int {
	res := make([]int, 0)
	datamap := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		datamap[nums[i]] = i
	}
	for i := 0; i < len(nums); i++ {
		val := target - nums[i]
		if _, ok := datamap[val]; !ok {
			continue
		} else {
			if datamap[val] == i {
				continue
			} else {
				res = append(res, i)
				res = append(res, datamap[val])
				break
			}
		}
	}
	return res
}
func main() {
	fmt.Println(twoSum([]int{3, 3}, 6))
}

/*
func reverse(x int) int {
	var res int
	for x != 0 {
		if temp := int32(res); (temp*10)/10 != temp {
			return 0
		}
		res = res*10 + x%10
		x = x / 10
	}
	return res
}
func main() {
	fmt.Println(reverse(-789))
	fmt.Println(reverse(-789 % 10))
}

/*
func threeSum(nums []int) [][]int {
	brr := make([][]int, 0)
	sort.Ints(nums)
	if len(nums) <= 0 {
		return brr
	}
	for i := 0; i < len(nums)-2; i++ {
		left, right := i+1, len(nums)-1
		if nums[i] > 0 {
			return brr
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		for left < right {
			if nums[i]+nums[left]+nums[right] == 0 {
				brr = append(brr, []int{nums[i], nums[left], nums[right]})
				for left < right && nums[left] == nums[left+1] {
					left++
				}
				for left < right && nums[right] == nums[right-1] {
					right++
				}
				left++
				right--
			} else if nums[i]+nums[left]+nums[right] > 0 {
				right--
			} else {
				left++
			}
		}
	}
	return brr
}
func main() {
	var crr []int = []int{-1, 0, 1, 2, -1, -4}
	fmt.Println(threeSum(crr))
}

/*
func divide(dividend int, divisor int) int {
	var count int = 0
	sign := false
	if (dividend ^ divisor) < 0 { // 两数任意一个为负数
		sign = true
	}
	if divisor == -2147483648 { // 除数边界值特殊处理
		if dividend == -2147483648 {
			return 1
		} else {
			return 0
		}
	}
	if dividend == -2147483648 { // 被除数边界值特殊处理
		if divisor == -1 {
			return 2147483647
		} else if divisor == 1 {
			return -2147483648
		}
		dividend += int(math.Abs(float64(divisor))) // 先执行一次加操作，避免abs转换溢出
		count++
	}
	var dvdnum int = int(math.Abs(float64(dividend)))
	var dvrnum int = int(math.Abs(float64(divisor)))
	for dvdnum >= dvrnum {
		val := 1
		su := dvrnum
		for su < (dvdnum >> 1) {
			su += su
			val += val
			fmt.Println(1000 + val)
			fmt.Println(100 + su)
		}
		count += val
		fmt.Println(100000 + count)
		dvdnum -= su
		fmt.Println(10000 + dvdnum)
	}
	if sign == true {
		return 0 - count
	} else {
		return count
	}
}
func main() {
	fmt.Println(divide(20, 3))
}

/*
func numIslands(grid [][]string) int {
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == "1" {
				dfs(grid, i, j)
				count++
			}
		}
	}
	return count
}
func dfs(grid [][]string, i, j int) {
	if (i < 0 || j < 0) || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == "0" {
		return
	}
	grid[i][j] = "0"
	dfs(grid, i+1, j)
	dfs(grid, i-1, j)
	dfs(grid, i, j-1)
	dfs(grid, i, j+1)
}

func main() {
	var arr [][]string = [][]string{
		{"1", "1", "0", "0", "0"},
		{"1", "1", "0", "0", "0"},
		{"0", "0", "1", "0", "0"},
		{"0", "0", "0", "1", "1"}}
	fmt.Println(numIslands(arr))
}
func IntegerToString(m int) string { //int转string
	s := strconv.Itoa(m)
	return s
}
func StringToInteger(s string) int { //string转int
	m, _ := strconv.Atoi(s)
	return m
}

/*
func singleNumber(nums []int) int {
	datamap := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		if _, ok := datamap[nums[i]]; ok {
			datamap[nums[i]] += 1
			for j := 0; j < len(nums); j++ {
				if nums[j] == nums[i] {
					nums[j] = 100000
				}
			}
		} else {
			datamap[nums[i]] = 1
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] != 100000 {
			return nums[i]
		}
	}
	return -1
}

func main() {
	var arr []int = []int{1, 2, 1, 1, 2, 2, 3, 4, 4, 4}
	fmt.Println(singleNumber(arr))
}

/*
func Process(ch chan int) {
	//Do some work...
	time.Sleep(time.Second)

	ch <- 1 //管道中写入一个元素表示当前协程已结束
}

func main() {
	channels := make([]chan int, 10) //创建一个10个元素的切片，元素类型为channel

	for i := 0; i < 10; i++ {
		channels[i] = make(chan int) //切片中放入一个channel
		go Process(channels[i])      //启动协程，传一个管道用于通信
	}

	for i, ch := range channels { //遍历切片，等待子协程结束
		<-ch
		fmt.Println("Routine ", i, " quit!")
	}
}

/*
func a() int {
	var i int = 0
	go func() {
		i++
	}()
	time.Sleep(5 * time.Second)
	return i
}
func b(i int) int {
	return i * 10
}
func main() {
	i := a()
	j := b(i)
	fmt.Println(i)
	fmt.Println(j)
}

/*
func permute(nums []int) [][]int {
	res := [][]int{}
	visited := map[int]bool{}

	var dfs func(path []int)
	dfs = func(path []int) {
		if len(path) == len(nums) {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for _, n := range nums {
			if visited[n] {
				continue
			}
			path = append(path, n)
			visited[n] = true
			dfs(path)
			path = path[:len(path)-1]
			visited[n] = false
		}
	}
	dfs([]int{})
	return res
}
func main() {
	var nums []int = []int{1, 2, 3}
	fmt.Println(permute(nums))
	fmt.Println(len(permute(nums)))
}

/*
func QuickSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}
	splitdata := arr[0]          //第一个数据
	low := make([]int, 0, 0)     //比我小的数据
	hight := make([]int, 0, 0)   //比我大的数据
	mid := make([]int, 0, 0)     //与我一样大的数据
	mid = append(mid, splitdata) //加入一个
	for i := 1; i < len(arr); i++ {
		if arr[i] < splitdata {
			low = append(low, arr[i])
		} else if arr[i] > splitdata {
			hight = append(hight, arr[i])
		} else {
			mid = append(mid, arr[i])
		}
	}
	low, hight = QuickSort(low), QuickSort(hight)
	myarr := append(append(low, mid...), hight...)
	return myarr
}

//快读排序算法
func main() {
	arr := []int{1, 9, 10, 30, 2, 5, 45, 8, 63, 234, 12}
	fmt.Println(QuickSort(arr))
}

/*
func main() {
	//定义命令行参数方式1
	var name string
	var age int
	var married bool
	var delay time.Duration
	flag.StringVar(&name, "name", "张三", "姓名")
	flag.IntVar(&age, "age", 18, "年龄")
	flag.BoolVar(&married, "married", false, "婚否")
	flag.DurationVar(&delay, "d", 0, "延迟的时间间隔")

	//解析命令行参数
	flag.Parse()
	fmt.Println(name, age, married, delay)
	//返回命令行参数后的其他参数
	fmt.Println(flag.Args())
	//返回命令行参数后的其他参数个数
	fmt.Println(flag.NArg())
	//返回使用的命令行参数个数
	fmt.Println(flag.NFlag())
}

/*
func main() {
	var a int = 0
	fmt.Scan(&a)
	fmt.Println(a + 20)

}

/*
func main() {
	str := "abroad cn"
	if strings.Index(str, "cn") < 0 {
		fmt.Println(str)
	} else {
		fmt.Println("00000")
	}
}

/*
func checkTopLevelDomain(domain string) int {
	if len(domain) == 0 {
		return -2
	}
	leftDot := strings.IndexByte(domain, '.')
	if leftDot == -1 {
		return -2
	}
	rightDot := strings.LastIndexByte(domain, '.')

	if leftDot == rightDot {
		return 0
	}
	rSecondDot := strings.LastIndexByte(domain[0:rightDot], '.')
	var wordFirst = domain[rightDot+1:]
	if wordFirst == "cn" {
		rThirdDot := strings.LastIndexByte(domain[0:rSecondDot], '.')
		if rThirdDot == -1 {
			return -1
		} else {
			return -1
		}

	}
	return -1
}

func GetTopLevelDomain(domain string) (string, int) {
	rc := checkTopLevelDomain(domain)
	if rc == -2 {
		return "", -1
	} else if rc == -1 {
		idxFirstDot := strings.LastIndexByte(domain, '.')
		idxSecondDot := strings.LastIndexByte(domain[0:idxFirstDot], '.')
		wordFirst := domain[idxFirstDot+1:]
		wordSecond := domain[idxSecondDot+1 : idxFirstDot]
		if wordFirst == "cn" {
			switch wordSecond {
			case "com", "org", "net", "gov", "sh", "bj", "edu", "mo", "hk", "tw", "tj", "cq", "he", "sx", "nm", "ln", "jl", "gd",
				"ac", "hl", "js", "zj", "ah", "fj", "jx", "sd", "ha", "hb", "hn", "gx", "hi", "sc", "gz", "yn", "xz", "sn", "gs",
				"qh", "nx", "xj":
				idxThirdDot := strings.LastIndexByte(domain[0:idxSecondDot], '.')
				if idxThirdDot != -1 {
					return domain[idxThirdDot+1:], 0
				}
				return domain, 0
			default:
				return domain[idxSecondDot+1:], 0
			}

		} else if wordFirst == "hk" || wordFirst == "tw" {
			if wordSecond == "com" {
				idxThirdDot := strings.LastIndexByte(domain[0:idxSecondDot], '.')
				if idxThirdDot != -1 {
					return domain[idxThirdDot+1:], 0
				}
				return domain, 0
			}
			return domain[idxSecondDot+1:], 0
		} else if wordFirst == "uk" || wordFirst == "in" || wordFirst == "jp" || wordFirst == "kr" {
			if wordSecond == "co" {
				idxThirdDot := strings.LastIndexByte(domain[0:idxSecondDot], '.')
				if idxThirdDot != -1 {
					return domain[idxThirdDot+1:], 0
				}
				return domain, 0
			}
			return domain[idxSecondDot+1:], 0
		} else if wordFirst == "com" {
			if wordSecond == "cn" {
				idxThirdDot := strings.LastIndexByte(domain[0:idxSecondDot], '.')
				if idxThirdDot != -1 {
					return domain[idxThirdDot+1:], 0
				}
				return domain, 0
			}
			return domain[idxSecondDot+1:], 0
		} else {
			return domain[idxSecondDot+1:], 0
		}

	} else {
		return domain, 0
	}
}
func main() {
	domain := "looper10000.ucloud.com.cn"
	topdomain, rc := GetTopLevelDomain(domain)
	fmt.Println(topdomain)
	fmt.Println(rc)
}

/*
func Search(n int, f func(int) bool) int {
	// Define f(-1) == false and f(n) == true.
	// Invariant: f(i-1) == false, f(j) == true.
	i, j := 0, n
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if !f(h) {
			i = h + 1 // preserves f(i-1) == false
		} else {
			j = h // preserves f(j) == true
		}
	}
	// i == j, f(i-1) == false, and f(j) (= f(i)) == true  =>  answer is i.
	return i
}
func main() {
	nums := []int64{1, 2, 3, 3, 8, 11, 27, 29}
	fmt.Println(Search(len(nums), func(a int) bool {
		return nums[a] >= 11
	}))
}

/*
var x int64
var l sync.Mutex
var wg sync.WaitGroup

// 普通版加函数
func add() {
	// x = x + 1
	x++ // 等价于上面的操作
	wg.Done()
}

// 互斥锁版加函数
func mutexAdd() {
	l.Lock()
	x++
	l.Unlock()
	wg.Done()
}

// 原子操作版加函数
func atomicAdd() {
	atomic.AddInt64(&x, 1)
	wg.Done()
}

func main() {
	start := time.Now()
	for i := 0; i < 10000; i++ {
		wg.Add(1)
		//go add() // 普通版add函数 不是并发安全的
		go mutexAdd() // 加锁版add函数 是并发安全的，但是加锁性能开销大
		//go atomicAdd() // 原子操作版add函数 是并发安全，性能优于加锁版
	}
	wg.Wait()
	end := time.Now()
	fmt.Println(x)
	fmt.Println(end.Sub(start))
}

/*

var m = make(map[string]int)

func get(key string) int {
	return m[key]
}

func set(key string, value int) {
	m[key] = value
}

func main() {
	wg := sync.WaitGroup{}
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(n int) {
			key := strconv.Itoa(n)
			set(key, n)
			fmt.Printf("k=:%v,v:=%v\n", key, get(key))
			wg.Done()
		}(i)
	}
	wg.Wait()
}

/*
var m = sync.Map{}

func main() {
	wg := sync.WaitGroup{}
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(n int) {
			key := strconv.Itoa(n)
			m.Store(key, n)
			value, _ := m.Load(key)
			fmt.Printf("k=:%v,v:=%v\n", key, value)
			wg.Done()
		}(i)
	}
	wg.Wait()
}

/*
var (
	x      int64
	wg     sync.WaitGroup
	lock   sync.Mutex
	rwlock sync.RWMutex
)

func write() {
	// lock.Lock()   // 加互斥锁
	rwlock.Lock() // 加写锁
	x = x + 1
	time.Sleep(10 * time.Millisecond) // 假设读操作耗时10毫秒
	rwlock.Unlock()                   // 解写锁
	// lock.Unlock()                     // 解互斥锁
	wg.Done()
}

func read() {
	// lock.Lock()                  // 加互斥锁
	rwlock.RLock()               // 加读锁
	time.Sleep(time.Millisecond) // 假设读操作耗时1毫秒
	rwlock.RUnlock()             // 解读锁
	// lock.Unlock()                // 解互斥锁
	wg.Done()
}

func main() {
	start := time.Now()
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go write()
	}

	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go read()
	}

	wg.Wait()
	end := time.Now()
	fmt.Println(end.Sub(start))
}

/*
var x int64
var wg sync.WaitGroup
var lock sync.Mutex

func add() {
	for i := 0; i < 5000; i++ {
		lock.Lock() // 加锁
		x = x + 1
		lock.Unlock() // 解锁
	}
	wg.Done()
}
func main() {
	wg.Add(2)
	go add()
	go add()
	wg.Wait()
	fmt.Println(x)
}

/*
type Job struct {
	// id
	Id int
	// 需要计算的随机数
	RandNum int
}

type Result struct {
	// 这里必须传对象实例
	job *Job
	// 求和
	sum int
}

func main() {
	// 需要2个管道
	// 1.job管道
	jobChan := make(chan *Job, 128)
	// 2.结果管道
	resultChan := make(chan *Result, 128)
	// 3.创建工作池
	createPool(64, jobChan, resultChan)
	// 4.开个打印的协程
	go func(resultChan chan *Result) {
		// 遍历结果管道打印
		for result := range resultChan {
			fmt.Printf("job id:%v randnum:%v result:%d\n", result.job.Id,
				result.job.RandNum, result.sum)
		}
	}(resultChan)
	var id int
	// 循环创建job，输入到管道
	for {
		id++
		// 生成随机数
		r_num := rand.Int()
		job := &Job{
			Id:      id,
			RandNum: r_num,
		}
		jobChan <- job
	}
}

// 创建工作池
// 参数1：开几个协程
func createPool(num int, jobChan chan *Job, resultChan chan *Result) {
	// 根据开协程个数，去跑运行
	for i := 0; i < num; i++ {
		go func(jobChan chan *Job, resultChan chan *Result) {
			// 执行运算
			// 遍历job管道所有数据，进行相加
			for job := range jobChan {
				// 随机数接过来
				r_num := job.RandNum
				// 随机数每一位相加
				// 定义返回值
				var sum int
				for r_num != 0 {
					tmp := r_num % 10
					sum += tmp
					r_num /= 10
				}
				// 想要的结果是Result
				r := &Result{
					job: job,
					sum: sum,
				}
				//运算结果扔到管道
				resultChan <- r
			}
		}(jobChan, resultChan)
	}
}

/*
func main() {
	ch1 := make(chan int)
	ch2 := make(chan int)
	// 开启goroutine将0~100的数发送到ch1中
	go func() {
		for i := 0; i < 100; i++ {
			ch1 <- i
		}
		close(ch1)
	}()
	// 开启goroutine从ch1中接收值，并将该值的平方发送到ch2中
	go func() {
		for {
			i, ok := <-ch1 // 通道关闭后再取值ok=false
			if !ok {
				break
			}
			ch2 <- i * i
		}
		close(ch2)
	}()
	// 在主goroutine中从ch2中接收值打印
	for i := range ch2 { // 通道关闭后会退出for range循环
		fmt.Println(i)
	}
}

/*
func recv(c chan int) {
	ret := <-c
	fmt.Println("接收成功", ret)
}
func main() {
	ch := make(chan int)
	go recv(ch) // 启用goroutine从通道接收值
	ch <- 10
	fmt.Println("发送成功")
}

/*
type Peopler interface {
	People()
	Woman()
}
type Kid struct {
	v int
}
type Man struct {
}

func (k Kid) People() {
	k.v = 100
	fmt.Println(3)
}
func (k Man) People() {
	fmt.Println(5)
}
func (m *Man) Woman() {
	fmt.Println("oooooooo")
}
func (k *Kid) Woman() {
	fmt.Println("tttttttt")
}
func main() {
	var p Peopler = &Kid{}
	p.People()

	p = &Man{}
	p.Woman()
}

/*
func kk() {
	var s [7]string
	m := make([]int, 0)
	for i := range s {
		defer func() []int {
			m = append(m, i)
			return m
		}()
	}
}
func main() {

	fmt.Println()
}

/*
func test01(base int) (func(int) int, func(int) int) {
	// 定义2个函数，并返回
	// 相加
	add := func(i int) int {
		base += i
		return base
	}
	// 相减
	sub := func(i int) int {
		base -= i
		return base
	}
	// 返回
	return add, sub
}

func main() {
	f1, f2 := test01(10)
	// base一直是没有消
	fmt.Println(f1(1), f2(2))
	// 此时base是9
	fmt.Println(f1(3), f2(4))
}

/*
// 外部引用函数参数局部变量
func add(base int) func(int) int {
		return func(i int) int {
			base += i
			return base
		}
}

func main() {
	tmp1 := add(10)
	fmt.Println(tmp1(1), tmp1(2))
	// 此时tmp1和tmp2不是一个实体了
	tmp2 := add(100)
	fmt.Println(tmp2(1), tmp2(2))
}

/*
func main() {
	s := make(map[int]int)
	m := []int{1, 2, 2, 3, 4, 1, 2, 2}
	for i := 0; i < len(m); i++ {
		s[m[i]] += m[i]
	}
	fmt.Println(s)
}

/*
func main() {
	re, _ := http.Get("https://uxiao.ucloudadmin.com/?ticket=ST-1827508-v1iz8a1-VMLriWRbKU7h9vzH5uUlocalhost#/api-manager/sections")
	body, _ := ioutil.ReadAll(re.Body)
	fmt.Println(string(body))
}

/*
var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	flag.Parse()
	f, err := os.Create(*cpuprofile)
	if err != nil {
		return
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()
}

/*
func main() {
	pattern := "[a-zA-Z]"
	result, _ := regexp.MatchString(pattern, str)
	fmt.Println(result)
}

/*
func main() {
	s := "2021-04-20T00:42:00Z"
	t, _ := time.Parse(time.RFC3339, s)
	loc, _ := time.LoadLocation("Local")
	te, _ := time.ParseInLocation("2006-01-02 15:04:05", t.Format("2006-01-02 15:04:05"), loc)
	fmt.Println(TimeToUnixHourMin(te.Format("2006-01-02 15:04:05")))
}
func TimeToUnixHourMin(s string) int64 { //时间转时间戳
	loc, _ := time.LoadLocation("Local")
	times, _ := time.ParseInLocation("2006-01-02 15:04:05", s, loc)
	timeunix := times.Unix()
	return timeunix
}

/*
var c = 10
func main(){
	fmt.Println(c)
}

/*
type A struct {
	B int
	C string
}

func M() *A {
	s := &A{}
	s.B = 10
	s.C = "uuuu"
	return s
}
func (t *A) Y() {
	v := t.B + 9
	n := t.C
	fmt.Println(v)
	fmt.Println(n)
}
func main() {
	s := M()
	s.Y()
}

/*
func main() {
	if find := strings.Contains("test-v1", "v1"); find {
		fmt.Println("find the character.")
	}
}

/*
func main() {
	var provlist []string = []string{"anhui", "beijing", "chongqing", "fujian", "gansu", "guangdong", "guangxi", "guizhou", "hainan", "hebei", "heilongjiang", "henan", "hubei", "hunan", "jiangsu", "jiangxi", "jilin", "liaoning", "neimenggu", "ningxia", "qinghai", "shananxi", "shandong", "shanghai", "shanxi", "sichuan", "tianjin", "xinjiang", "xizang", "yunnan", "zhejiang", "qita"}
	fmt.Println(len(provlist))
}
/*
http://api.ucloudadmin.com/?Action=GetUcdnMonthlyBandwidth&TopOrganizationId=56012663&ChargeMonth=1617206400&DomainIds.0=ucdn-g35hz4uf&DomainIds.0=ucdn-2qpotk2h&Area=cn&DataType=0
TopOrganizationId int 必传
OrganizationId  int 项目id，可选，不传查询整个公司id下的数据
DomainIds  []string 域名id，可选， 不传则查询整个账户或整个公司id下的域名（DomainIds.0=oooo&DomainIds.1=pppp)
ChargeMonth int64 月份时间戳，必传， 时间戳只要是某个月的时间戳就返回某个月的数据，例如传1615132800， 就查询整个三月份的数据
Area string 区域， 可选， 国内（cn）、国外（abroad）、全部区域不传或者（all）
DataType int 获取数据类型，0（概要数据） 不等于0（原始数据）
/*
func main() {
	mapOrderDetail := []map[string]interface{}{{"ProductId": 200007, "Multiple": 444444.444}}
	mapBodyData := map[string]interface{}{"Action": "GetBuyPrice", "Backend": "UBill", "TopOrganizationId:": 7777, "OrganizationId": 777777, "ProductType:": 7, "RegionId": 1, "ChargeType": 102, "Quantity": 1, "Count": 1, "OrderDetail": mapOrderDetail, "Channel": 0}

	jsonStr, err := json.Marshal(mapBodyData)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(jsonStr))
}

/*
func main() {
	var tt int64 = 1512230400
	ts := time.Unix(tt, 0)
	tst := ts.AddDate(0, 1, 0)
	fmt.Println(ts)
	fmt.Println(tst)
}

/*
func main() {
	t, _ := time.Parse(time.RFC3339, "2021-04-09T06:50:02Z")
	loc, _ := time.LoadLocation("Local")
	te, _ := time.ParseInLocation("2006-01-02 15:04:05", t.Format("2006-01-02 15:04:05"), loc)
	fmt.Println(te.Format("2006-01-02 15:04:05"))

}

/*
func main() {
	t, _ := time.Parse(time.RFC3339, "2021-03-22T08:23:42Z")
	fmt.Println(t.Format("2006-01-02 15:04:05"))
	loc, _ := time.LoadLocation("Local")
	times, _ := time.ParseInLocation("2006-01-02 15:04:05", "2021-03-22 08:23:42", loc)
	timeunix := times.Unix()
	fmt.Println(timeunix)
}

/*
func main() {
	record := 0
	var sortarr = []int{1, 1, 3, 4, 7, 9, 9}
	for j := 0; j < len(sortarr); j++ {
		if j > 0 {
			if sortarr[j]-record != 1 {
				sortarr = append(sortarr, 0)
				copy(sortarr[j+1:], sortarr[j:])
				sortarr[j] = record + 1
				record = record + 1
				continue
			}
		}
		record = sortarr[j]
	}
	fmt.Println(sortarr)
}

/*
func main() {
	var a = []int{1, 2, 4, 5, 6, 9}
	a = append(a, 0)
	fmt.Println(a[2:])
	fmt.Println(a[3:])
	copy(a[3:], a[2:])
	a[2] = 3

	fmt.Println(a)
}

/*
func main() {
	var arr []string = []string{"A", "a", "b", "c", "a", "b"}
	datamap := make(map[string]map[int]int)
	for i := 0; i < len(arr); i++ {
		if _, ok := datamap[arr[i]]; !ok {
			datamap[arr[i]] = make(map[int]int)
		}
	}
	var brr []int = []int{1, 2, 2, 2, 3, 4, 5}
	for k := range datamap {
		for i := 0; i < len(brr); i++ {
			datamap[k][brr[i]] = brr[i]
		}
	}
	fmt.Println(datamap)
}

/*
func UnixToTime(m int64) string { //时间戳转时间
	formatTimeStr := time.Unix(m, 0).Format("2006-01-02 15:04:05")
	return formatTimeStr
}
func StringToInteger64(s string) int64 { //string转int64
	m, _ := strconv.ParseInt(s, 10, 64)
	return m
}
func main() {
	s := "1616653416"
	m := UnixToTime(StringToInteger64(string(s)))
	t, _ := time.ParseInLocation("2006-01-02 15:04:05", m, time.Local)
	create_time := t.UTC().Format("2006-01-02T15:04:05.000+08:00")
	fmt.Println(string(create_time))
}

/*
func CheckDomainLegal(Domain string) bool {
	Rescode := false
	flag := false
	StrArr := strings.Split(Domain, ".")
	if len(StrArr) < 2 {
		return false
	}
	for i:=0; i<len(StrArr); i++ {
		flag, _ = regexp.MatchString("^[A-Za-z0-9]+$", StrArr[i])
		if flag == false {
			for j:=0; j<len(StrArr[i]); j++ {
				if string(StrArr[i][j]) != "-" && string(StrArr[i][j]) != "_" {
					sign, _ := regexp.MatchString("^[A-Za-z0-9]+$", string(StrArr[i][j]))
					if sign == false {
						Rescode = false
						break
					}else{
						Rescode = true
					}
				}else{
					if j == len(StrArr[i]) {
						Rescode =true
					}
				}
			}
		}else{
			Rescode = true
		}
		if Rescode == false {
			break
		}
	}
	return Rescode
}
func main() {
	fmt.Println(CheckDomainLegal(".lll.ccc.mmm"))
	fmt.Println(CheckDomainLegal("1.L"))
	fmt.Println(CheckDomainLegal("ppp_ooo.kc"))
	fmt.Println(CheckDomainLegal("ppp-ooo.kc"))

}

/*
func main() {
	a := 1
	b := 10
	fmt.Println(a&b)
}

/*
func f() (res int) {
	i := 1
	defer func() {
		res ++
	}()
	return i
}
/*
func f() (r int) {
    defer func(r int) {
        r = r + 5
    }(r)
    return 1
}
/*
func f() (r int) {
    t := 5
    defer func() {
		t = t + 5
		fmt.Println("ttttttt")
		fmt.Println(t)
	}()
	fmt.Println("aaaaaaaaaaa")
	fmt.Println(t)
    return t
}
/*
func f() (r int) {
    t := 5
    r = t
    defer func() {
		t = t + 5
		fmt.Println("tttttttttt")
		fmt.Println(t)
	}
	fmt.Println(t)
    return
}
func main() {
	a := f()
	fmt.Println(a)
}

/*
type a struct {
	B []map[string]interface{}
	C string
}
func My() []a {
	var m []a = []a{}
	in := a{}
	arr := make([]map[string]interface{}, 0)
	data := make(map[string]interface{})
	data["h"] = "ooo"
	data["u"] = 100
	arr = append(arr, data)

	data = make(map[string]interface{})
	data["h"] = "lllll"
	data["u"] = 0
	arr = append(arr, data)
	in.B = arr
	in.C = "cccccc"
	m = append(m, in)

	arr = make([]map[string]interface{}, 0)
	data = make(map[string]interface{})
	data["h"] = "yyyyy"
	data["u"] = 88
	arr = append(arr, data)
	data = make(map[string]interface{})
	data["h"] = "uuuuuuu"
	data["u"] = 66
	arr = append(arr, data)
	in.B = arr
	in.C = "qqqqqq"
	m = append(m, in)
	return m
}
func main() {
	m := My()
	//var CdnBandwidth int = 0
	for i:=0; i<len(m); i++ {
		for j:=0; j<len(m[i].B); j++ {
			CdnBandwidth, ok := m[i].B[j]["u"].(int)
			if !ok {
				fmt.Println("oooooooooooooooooooooooooooooooooooo")
			}
			fmt.Println(CdnBandwidth)
		}
		fmt.Println(m[i].B)
		fmt.Println(m[i].C)
	}
}

/*
type EdgeRoot struct {				//边缘配置
	XMLName					xml.Name			`xml:"root"`
	Domain					string				`xml:"domain"`
	DomainId				int64				`xml:"domain_id"`
	OriginHost				string				`xml:"origin_host"`
	SpecialConfigs			SpecialConfigEdge	`xml:"special_configs"`
}
type SpecialConfigEdge struct {
	CacheRules				CacheRuleInfo		`xml:"cache_rules"`
	CacheKey				CacheKeyInfo		`xml:"cache_key"`
	Refer					ReferInfo			`xml:"refer"`
	IpControl				IpControlInfo		`xml:"ipcontrol"`
	HostForcache			string				`xml:"host_forcache"`
	HeaderWriter			HeaderWriterInfo	`xml:"header_writer"`
	RegexRules				RegexRulesInfo		`xml:"regex_rules"`
}
type CacheRuleInfo struct {
	Rules					[]CacheRuleContent	`xml:"rules>rule"`
	Enable					int					`xml:"enable"`
}
type CacheRuleContent struct{
	UrlPatton				string				`xml:"url_patton"`
	HttpCode				int					`xml:"httpcode"`
	TTL						int					`xml:"ttl"`
	IgnoreCacheControl		int					`xml:"ignore-cache-control"`
}
type CacheKeyInfo	struct {
	Rules					[]KeyContent		`xml:"rules>rule"`
	Enable					int					`xml:"enable"`
	Patton					string				`xml:"patton"`
}
type KeyContent struct {
	Patton					string				`xml:"patton"`
	Key						string				`xml:"key"`
}
type ReferInfo struct {
	Enable					int					`xml:"enable"`
	Allownullreferer		int					`xml:"allownullreferer"`
	ValidList				[]string			`xml:"valid_list"`
}
type IpControlInfo	struct {
	Enable					int					`xml:"enable"`
	InvalidList				[]string			`xml:"invalid_list"`
}
type HeaderWriterInfo struct {
	Enable					int					`xml:"enable"`
	ClientRspSet			string				`xml:"client_rsp_set"`
	OriginreqSet			string				`xml:"origin_req_set"`
}
type RegexRulesInfo	struct {
	Rules					RegexRulesRulesInfo	`xml:"rules"`
}
type RegexRulesRulesInfo struct {
	Rule					[]string			`xml:"rule"`
}
func main(){
	Root := EdgeRoot{
		Domain : "pppp.ccc",
		DomainId : 1,
		OriginHost : "kkkk",
		SpecialConfigs : SpecialConfigEdge{
			CacheRules : CacheRuleInfo{
				Rules : []CacheRuleContent{
					CacheRuleContent{
						UrlPatton : "ssss",
						HttpCode  : 1,
						TTL : 2,
						IgnoreCacheControl : 3,
					},
					CacheRuleContent{
						UrlPatton : "qqqqqqqqqqqqqqqq",
						HttpCode  : 800,
						TTL : 900,
						IgnoreCacheControl : 100,
					},
				},
				Enable : 1,
			},
			CacheKey : CacheKeyInfo{
				Rules : []KeyContent{
					KeyContent{
						Patton : "pppp",
						Key : "llll",
					},
					KeyContent{
						Patton : "pppp",
						Key : "llll",
					},
				},
				Enable : 1,
				Patton : "iiiiii",
			},
			Refer : ReferInfo{
				Enable : 5,
				Allownullreferer : 10,
				ValidList : []string{"rrr","tttt"},
			},
			IpControl : IpControlInfo{
				Enable : 2,
				InvalidList : []string{"aaaaa","eeee"},
			},
			HostForcache : "mmmmmmmmmmmmmmmmm",
			HeaderWriter : HeaderWriterInfo{
				Enable : 800,
				ClientRspSet : "lll",
				OriginreqSet : "pppp",
			},
			RegexRules : RegexRulesInfo {
				Rules : RegexRulesRulesInfo{
					Rule : []string{"ttttttttttt","jjjjjjjjjjjjjjjjj"},
				},
			},
		},
	}
	v, err := xml.MarshalIndent(Root, "", "         ")
    if err != nil {
        fmt.Println("marshal xml value error, error msg:%s", err.Error())
    }

    fmt.Println("marshal xml value", string(v))
}



/*
type Recurlyservers struct {//后面的内容是struct tag，标签，是用来辅助反射的
    XMLName xml.Name `xml:"servers"` //将元素名写入该字段
    Version string `xml:"version,attr"` //将version该属性的值写入该字段
    Svs []server `xml:"server"`
    Description string `xml:",innerxml"` //Unmarshal函数直接将对应原始XML文本写入该字段
}

type server struct{
    XMLName xml.Name `xml:"server"`
    ServerName string `xml:"serverName"`
    ServerIP []string `xml:"serverIP"`
}
func main() {
	data:=`<?xml version="1.0" encoding="utf-8"?>
	<servers version="1">
		<server>
			<serverName>Shanghai_VPN</serverName>
			<serverIP>127.0.0.1</serverIP>
			<serverIP>1.1.1.1</serverIP>
		</server>
		<server>
			<serverName>Beijing_VPN</serverName>
			<serverIP>127.0.0.2</serverIP>
			<serverIP>2.2.2.2</serverIP>
		</server>
	</servers>`
	bs := Recurlyservers{}
 	xml.Unmarshal([]byte(data), &bs)

	fmt.Println(bs.XMLName)
	fmt.Println(bs.Svs[0].ServerName)
	fmt.Println(bs.Svs[0].ServerIP[0])
	fmt.Println(bs.Svs[0].ServerIP[1])
	fmt.Println(bs.Svs[1].ServerIP[0])
	fmt.Println(bs.Svs[1].ServerIP[1])
	fmt.Println(bs.Version)
	fmt.Println(bs.Description)
}
/*
var R int = 10
func main() {
	ch := make(chan bool,10)
	for i:=0; i<R; i++ {
		go func(s int){
			Ta(s)
			//fmt.Println("ooo",s)
			ch <- true
		}(i)
	}
	for i:=0; i<R; i++ {
		k := <-ch
		fmt.Println(k)
	}
}

func Ta(a int) {
	fmt.Println("ppp", a)
}
func Workers(task func(int)) chan int {
	in := make(chan int)
	for i:=0; i<10; i++ {
		go func(){
			for {
				v, ok := <-in
				if ok {
					task(v)
				}else{
					return
				}
			}
		}()
	}
	return in
}
func main() {
	ack := make(chan bool, 10)
	wr := Workers(func(a int) {
		Ta(a)
		ack <- true
	})
	for i:=0; i<10; i++ {
		wr <- i
	}
	for i:=0; i<10; i++ {
		<-ack
	}
}
/*
func CheckDomainLegal(Domain string) int {
	Rescode := false
	flag := false
	StrArr := strings.Split(Domain, ".")
	if len(StrArr) < 2 {
		return -1
	}
	for i:=0; i<len(StrArr); i++ {
		flag, _ = regexp.MatchString("^[A-Za-z0-9]+$", StrArr[i])
		if flag == false {
			for j:=0; j<len(StrArr[i]); j++ {
				if string(StrArr[i][j]) != "-" && string(StrArr[i][j]) != "_" {
					sign, _ := regexp.MatchString("^[A-Za-z0-9]+$", string(StrArr[i][j]))
					if sign == false {
						Rescode = false
						break
					}else{
						Rescode = true
					}
				}else{
					if j == len(StrArr[i]) {
						Rescode =true
					}
				}
			}
		}else{
			Rescode = true
		}
		if Rescode == false {
			break
		}
	}
	m := 0
	if Rescode == true {
		m = 1
	}else{
		m = -1
	}
	return m
}
func main() {
	a := CheckDomainLegal("abc.com.cn")
	b := CheckDomainLegal("8-8pp.com.cn")
	c := CheckDomainLegal("7p*sp.com.cn")
	d := CheckDomainLegal(".cn")
	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(c)
	fmt.Println(d)
}
/*
func main(){
	DomainMap := "abc.com.cn"
	f := strings.Replace(DomainMap, ".", "0", -1)
	fmt.Println(f)
}


/*

func main(){
	a:="899999998"
	s:= ""
	for i:=0;i<len(a); i++ {
		if a[i]=='8' {
			s += "0"
		}
		s+=string(a[i])
	}
	fmt.Println(s)
}

func main() {
	PathPattern := "     888    999   pppppp   "
	fmt.Println(len(PathPattern))
	PathPattern = strings.TrimSpace(PathPattern)
	fmt.Println(PathPattern)
	fmt.Println(len(PathPattern))
}






/*
type t struct {
	A string
	B *bool
}
func main() {
	t := t{}
	t.A = "ppp"
	if t.B == nil {
		fmt.Println(t.A)
		fmt.Println("iiii")
	}
	t.B = false
	fmt.Println(*t.B)
}
/*
func main() {
	m := fmt.Sprintf("%x", md5.Sum([]byte("oooooooooooooooo"+"qqqqqqqqqqqqqqqqqqqqqq")))
	fmt.Println(m)
}
/*
type t struct {
	A string
	B *bool
}
func main() {
	t := t{}
	t.A = "ppp"
	if t.B == nil {
		fmt.Println(t.A)
		fmt.Println("iiii")
	}
	*t.B = false
	fmt.Println(*t.B)
}

/*
func main() {
	s := "[{\"a\":\"ll\",\"b\":\"pp\"},{\"a\":\"yy\",\"b\":\"cc\"}]"
	L := make([]*t, 0)
	err := json.Unmarshal([]byte(s), &L)
	if err == nil {
		fmt.Println(string(L[0].A))
	}else{
		fmt.Println(err)
	}
}



/*
func main() {
    runtime.GOMAXPROCS(1)
    wg := sync.WaitGroup{}
    wg.Add(20)
    for i := 0; i < 10; i++ {
        go func() {
            fmt.Println("A: ", i)
            wg.Done()
        }()
    }
    for i := 0; i < 10; i++ {
        go func(i int) {
            fmt.Println("B: ", i)
            wg.Done()
        }(i)
    }
    wg.Wait()
}
/*
func main(){
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT)
	signal.Notify(ch, syscall.SIGKILL)
	signal.Notify(ch, syscall.SIGTERM)
	for {
		go func(){
			c := <-ch
			if c == nil {
				os.Exit(0)
			}
			num()
			time.Sleep(2*time.Second)
		}()
		time.Sleep(1*time.Second)
	}
}
type a struct {
	C string
	F int
	B float64
	G string
}
func num() {
	l := make([]a,0)
	for i:=0; i<10; i++ {
		m := a{}
		m.C="ooooo"
		m.B=float64(i)
		l = append(l,m)
	}
	fmt.Println(l)
}
/*
func main() {
    // 1 定义当前进程PID文件
    sigfile := "./cli_syncStaffs.pid"

    _, err := os.Stat(sigfile)
    if err == nil {
        //pid文件存在-进程已经存在
        fmt.Println("PID file exist.running...")
        os.Exit(0)
    }

    // 2 创建当前进程的pid文件
    pidFileHandle, err := os.OpenFile(sigfile, os.O_RDONLY|os.O_CREATE, os.ModePerm)
    if err != nil {
        panic(err)
    }

	// 执行业务逻辑
	for {
		fmt.Println(time.Now().Format("2006-01-02 15:04:05"))
		fmt.Println(time.Now().Format("2006-01-02 15:04:05"))
	}
    // 执行完毕
    err= pidFileHandle.Close()
    if err!=nil {
        fmt.Println(err)
    }
    // 删除该文件
    err = os.Remove(sigfile)
    if err!=nil {
        fmt.Println(err)
    }
}

/*

var wg sync.WaitGroup

func main() {
    for {
		fmt.Println("start ")
		wg.Add(1)
		go test() //启动10个goroutine 来计算
    }
    // 阻塞-保证子协程跑完
    wg.Wait()
}

func test(){
    defer wg.Done()
    // 1 定义当前进程PID文件
    sigfile := "./cli_syncStaffs.pid"
    // 1 获取当前的pid文件(没有就自动创建)
    pidFileHandle, err := os.OpenFile(sigfile, os.O_RDONLY|os.O_CREATE, os.ModePerm)
    if err != nil {
        fmt.Println("open fail.")
        fmt.Println(err)
        return
    }
    defer pidFileHandle.Close()

    // 2 文件加锁
    err = syscall.Flock(int(pidFileHandle.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
    if err != nil {
        fmt.Println(err)
        fmt.Println("running...")
        return
    }
    defer syscall.Flock(int(pidFileHandle.Fd()), syscall.LOCK_UN)
    // 执行业务逻辑
    fmt.Println(time.Now().Format("2006-01-02 15:04:05"))
    time.Sleep(5*time.Second)
    fmt.Println(time.Now().Format("2006-01-02 15:04:05"))
    return
}


/*
func Init() {
    iManPid := fmt.Sprint(os.Getpid())
    tmpDir := os.TempDir()

    if err := ProcExsit(tmpDir); err == nil {
        pidFile, _ := os.Create(tmpDir + "\\imanPack.pid")
        defer pidFile.Close()

        pidFile.WriteString(iManPid)
    } else {
        os.Exit(1)
    }
}

// 判断进程是否启动
func ProcExsit(tmpDir string) (err error) {
    iManPidFile, err := os.Open(tmpDir + "\\imanPack.pid")
    defer iManPidFile.Close()
    if err == nil {
        filePid, err := ioutil.ReadAll(iManPidFile)
        if err == nil {
            pidStr := fmt.Sprintf("%s", filePid)
            pid, _ := strconv.Atoi(pidStr)
            _, err := os.FindProcess(pid)
            if err == nil {
                return errors.New("[ERROR] iMan升级工具已启动.")
            }
        }
    }
    return nil
}
func main() {
	Init()
	for {
		go num()
	}
	time.Sleep(2*time.Second)
}
/*
func FloatToStringTwo(f float64) string {
	value := strconv.FormatFloat(f, 'f', 5, 64)
	return value
}
func main(){
	var i =1.8889
	fmt.Println(FloatToStringTwo(i))
}
/*
func counter(out chan<- int) {
	for x := 0; x < 100; x++ {
		out <- x
 	}
 	close(out)
}
func squarer(out chan<- int, in <-chan int) {
 	for v := range in {
 		out <- v + v
 	}
 	close(out)
}
func printer(in <-chan int) {
 	for v := range in {
 		fmt.Println(v)
 	}
}
func main() {
 	naturals := make(chan int)
 	squares := make(chan int)
 	go counter(naturals)
 	go squarer(squares, naturals)
 	printer(squares)
}


/*
func main() {
	m := make(map[string]map[int]map[int]int)
	m["a"]=make(map[int]map[int]int)
	m["a"][8] = make(map[int]int)
	m["a"][8][2]=3
	m["a"][8][3]=5
	fmt.Println(m)
}


/*
func main() {
	var m []int = []int{}
	a := make([]int,0)
	fmt.Println(a)
	fmt.Println([0]int{})
	fmt.Println(m)
}
/*
func main() {
	var a []int = []int{1,2,5,8,3,5,64,1}
	for _,k := range a {
		fmt.Println(k)
	}
}
/*
func StringToInteger(s string) int { //string转int
	m, _ := strconv.Atoi(s)
	return m
}
func DecideOriginIpList(OriginIpList []string) (bool){
	if len(OriginIpList) < 0{
		return false
	}
	sign := true
	flag := true
	if len(OriginIpList) > 1 {
		flag = true
	}else{
		for i := 0; i<len(OriginIpList); i++ {
			value := net.ParseIP(OriginIpList[i])
			if value == nil {
				flag = false
				break
			}
		}
		tmp := len(OriginIpList[0])
		for i:=0; i<len(OriginIpList[0]); i++ {
			value := StringToInteger(fmt.Sprintf("%d", OriginIpList[0][i]))
			if (value<48 || value>57) && value != 46 {
				tmp --
			}
		}
		if tmp == len(OriginIpList[0]) {
			flag = true
		}
	}
	if flag == false {
		for i := 0; i<len(OriginIpList[0]); i++ {
			value := StringToInteger(fmt.Sprintf("%d", OriginIpList[0][i]))
			if i == 0 {
				if value>123 || (value<65 && value >57) || (value<97 && value>90) || value < 48 {
					sign = false
				}
				continue
			}
			if value != 46 && (value>123 || (value<65 && value >57) || (value<97 && value>90) || value < 48)  && value != 45 {
				sign = false
			}
		}
	}else{
		for i := 0; i<len(OriginIpList); i++ {
			value := net.ParseIP(OriginIpList[i])
			if value == nil {
				sign = false
				break
			}
		}
	}
	return sign
}
func main(){
	 a := []string{"1.1.1.1","192.168.13.8","172.23.7.249"}
	 b := []string{"4b-3dsd.ikl.com.cn"}
	 c := []string{".lll.ppp.com.cn"}
	 d := []string{"257.23.268.24"}
	 fmt.Println(DecideOriginIpList(a))
	 fmt.Println(DecideOriginIpList(b))
	 fmt.Println(DecideOriginIpList(c))
	 fmt.Println(DecideOriginIpList(d))


}

/*
func main(){
	a := "15abc1"
	for i:= 0; i<len(a); i++ {
		s:=fmt.Sprintf("%d",a[i])
		fmt.Println(s)
	}
}

/*
func IntegerToString(m int) string{ //int转string
	s := strconv.Itoa(m)
	return s
}
func StringToInteger(s string) int { //string转int
	m, _ := strconv.Atoi(s)
	return m
}

func numf(num []int) int {
	var maxvalue int
	var arr []int = []int{}
	if len(num)>0 {
		maxvalue = num[0]
	}
	for i:=0; i<len(num); i++ {
		if maxvalue < num [i] {
			maxvalue = num[i]
		}
		arr = append(arr,num[i])
	}
	fmt.Println(maxvalue)
	m := 10
	var a int = 1
	var j int = 1
	for ;j<10;j++{
		if maxvalue/a < 10 {
			break
		}else{
			a *= m
		}
	}
	fmt.Println(j)
	var index int = 1
	for i := 1; i<=j; i++{
		index *= m
	}
	index -= 1
	for i:=0; i<len(num);i++{
		s := num[i]%10
		for k :=0;k<j;k++{
			num[i] = num[i]*10+s
		}
	}
	var brr []int =	[]int{}
	for i:=0;i<len(num);i++{
		brr = append(brr,num[i])
	}
	fmt.Println(num)
	var str string = ""
	sort.Ints(num)
	fmt.Println(num)
	for i:=0; i<len(num);i++{
		for k:=0;k<len(brr);k++{
			if num[i] == brr[k] {
				str+=IntegerToString(arr[k])
			}
		}
	}
	return StringToInteger(str)
}
func main(){
	var m []int = []int{1,23,25,9,31}
	fmt.Println(numf(m))
}

/*
func numf(num []int) int {
	var value int = 1
	var maxvalue int
	var minvalue int
	if len(num) > 0 {
		maxvalue = num[0]
		minvalue = num[0]
	}else{
		return value
	}
	data := make(map[int]int)
	for i:=0; i<len(num); i++ {
		if num[i] > maxvalue {
			maxvalue = num[i]
		}
		if num[i] < minvalue {
			minvalue = num[i]
		}
		data[num[i]] = num[i]
	}
	if maxvalue <= 0 {
		maxvalue = 2
	}else{
		maxvalue += 2
	}
	if minvalue > 0 {
		minvalue = 0
	}
	for i := minvalue; i < maxvalue; i++{
		if _,ok := data[i]; !ok {
			if i > 0{
				value = i
				break
			}
		}
	}
	return value
}
func main(){
	var a []int = []int{7,8,9,11,12}
	fmt.Println(numf(a))
}


/*

		fmt.Printf("%s",str[1])
}
/*
func main() {
	var data map[string]interface{}
	data = map[string]interface{}{
		"a" : 8,
		"b" :"ppp",
		"c" : 5.5,
	}
	m,_ := json.Marshal(data)
	fmt.Println(string(m))
}
*/
/*
func a(o int, m ... interface{}){
	fmt.Println(m)
	fmt.Println(o)
}
func main(){
	a(8, "pppp","ooo")
}
/*	a(8,5)
	a(23,9.2)
	a(5,true)
	s := []int{1,2,5,3,6}
	a(7,s)
}

/*
type c struct {
	w int
	r int
}
type a struct{
	c
	q int
	str string
}
type b struct{
	a
	s string
	p int
}
func main() {
	m := b{}
	m.s = "oooo"
	m.p = 8
	m.a.q = 2
	m.a.str = "uuuuu"
	m.a.c.w = 78
	m.a.c.r = 123
	fmt.Println(m)
//	fmt.Println(a)
//	fmt.Println(b)
}
/*
/*
func QuickSortData(data []float64, start int, end int) ([]float64) {
    if (start < end) {
        base := data[start]
        left := start
        right := end
        for left < right {
            for left < right && data[right] >= base {
                right--
            }
            if left < right {
                data[left] = data[right]
                left++
            }
            for left < right && data[left] <= base {
                left++
            }
            if left < right {
                data[right] = data[left]
                right--
            }
        }
        data[left] = base
        QuickSortData(data, start, left - 1)
        QuickSortData(data, left + 1, end)
    }
    return data
}
func main(){
	m := []float64{1.2,2.5,0.2,0.6,0.1,8.5,7.3}
	a := QuickSortData(m,0,len(m)-1)
	fmt.Println(a)
}
/*
func QuickSort(Screen []map[string]interface{},start int, end int) ([] map[string]interface{}){
    if (start < end) {
        zero, _ := Screen[0]["Time"].(int64)
		base := zero
        value := Screen[start]
        left := start
        right := end
        for left < right {
			rightvalue, _ := Screen[right]["Time"].(int64)
            for left < right && rightvalue >= base {
                right--
            }
            if left < right {
                Screen[left] = Screen[right]
                left++
			}
			leftvalue, _ := Screen[left]["Time"].(int64)
            for left < right && leftvalue <= base {
                left++
            }
            if left < right {
                Screen[right] = Screen[left]
                right--
            }
        }
        Screen[left] = value
        QuickSort(Screen, start, left - 1)
        QuickSort(Screen, left + 1, end)
    }
    return Screen
}
/*
func QuickSortTime(BandwidthList []map[string]interface{}, start int, end int) ([] map[string]interface{}){
    if (start < end) {
		base,_ := BandwidthList[0]["Time"].(int64)
        value := BandwidthList[start]
        left := start
        right := end
        for left < right {
			Timeright, _ := BandwidthList[right]["Time"].(int64)
            for left < right && Timeright >= base {
                right--
            }
            if left < right {
                BandwidthList[left] = BandwidthList[right]
                left++
			}
			Timeleft, _ := BandwidthList[left]["Time"].(int64)
            for left < right && Timeleft <= base {
                left++
            }
            if left < right {
                BandwidthList[right] = BandwidthList[left]
                right--
            }
        }
        BandwidthList[left] = value
        QuickSortTime(BandwidthList, start, left - 1)
		QuickSortTime(BandwidthList, left + 1, end)
    }
    return BandwidthList
}*/
/*
func main(){
	var arr [] map[string]interface{}
	a := make(map[string]interface{})
	b := make(map[string]interface{})
	c := make(map[string]interface{})
	a = map[string]interface{}{
		"Time":8,
		"u":5.2,
	}
	b =map[string]interface{}{
                "Time":18,
                "u":5.2,
        }
	c=map[string]interface{}{
                "Time":2,
                "u":5.2,
        }
	arr = append(arr,a)
	arr = append(arr,b)
	arr = append(arr,c)
	k := QuickSort(arr,0,len(arr))
	fmt.Println(k)
}
/*
	var u []string = []string{"1","2"}
	c := map[string]interface{}{
		"a":u,
		"b":2,
		"m":"8",
	}
	p,_:=json.Marshal(c)
	list :=`curl -d '`+string(p)+`' http://172.18.181.2:8001/api/v3/statistics/aggregations`;

	fmt.Println(list)
}
/*
func m(a string) (base.ApiBaseResponse) {
	b,err := strconv.Atoi(a)
	if err!= nil {
		return ErrorResponse("SQL_OP_ERROR","sql op error")
	}
	return ErrorResponse("SQL_OP_ERROR","sql op error")
}
func main(){
	var a string = "5"
	b := m(a)
	Println(b)
}
/*
func QuickSortData(data []int, start int, end int) ([]int) {
    if (start > end) {
        base := data[start]
        value := data[start]
        left := start
        right := end
        for left > right {
            for left > right && data[right] <= base {
                right--
            }
            if left > right {
                data[left] = data[right]
                left++
            }
            for left > right && data[left]>= base {
                left++
            }
            if left > right {
                data[right] = data[left]
                right--
            }
        }
        data[left] = value
        QuickSortData(data, start, left - 1)
        QuickSortData(data, left + 1, end)
    }
    return data
}
func main(){
	var a []int= []int{1,5,2,3,6,8,9,7,1,2,3}
	b := QuickSortData(a,0,len(a)-1)
	fmt.Println(b)
}
/*
func main(){
	a := make([][2]int,0)
	a[0][0]=8
	a[0][1]=2
	fmt.Println(a)
}
/*
type a struct {
	b int
	c []map[string]int
}
func main(){
	f := []a{}
	var r []int = []int{5,4,2,3,9,5,6,4,7,5,8}
	for i:=0;i<len(r);i++{
		for j:=i+1;j<len(r);j++{
			if r[i]==r[j]{
				r[j]=0
			}
		}
	}
	var d []int = []int{}
	for i:=0;i<len(r);i++{
		if r[i]!=0{
			d=append(d,r[i])
		}
	}
	for i:=0;i<len(d);i++ {
		p := a{}
		var data []map[string]int
		for j:=0;j<len(r);j++{
			m := make(map[string]int)
			if d[i]==r[j]{
				m[strconv.Itoa(j)] = r[j]
				data=append(data,m)
			}
		}
		p.b=d[i]
		p.c=data
		f = append(f,p)
		fmt.Println(f)
	}
}

/*
func Unwrap(num int64, retain int) float64 { //int64转float64
	return float64(num) / math.Pow10(retain)
}
func main(){
	var a int64 = 12088
	fmt.Println(Unwrap(a,2)*100)
}

/*
func main() {
	var Record []string = []string{"5","6","p","6","5","0","p"}
	for i := 0; i<len(Record); i++ {
		for j := i+1 ; j<len(Record); j++ {
			if Record[i] == Record[j] {
				Record[j] = "0"
			}
		}
	}
	var b []string= []string{}
	for i:=0;i<len(Record);i++{
		if Record[i]!="0"{
			b = append(b,Record[i])
		}
	}
	var c = 0.0
	fmt.Println(b)
	fmt.Println(c)
}

/*

	var a string=""
	b,_:=strconv.Atoi(a)
	c := b+1
	fmt.Println(c)
}



/*
func FloatToStringTwo(f float64) string { //float64四舍五入保留两位小数输出字符串
	f1 := math.Trunc(f*1e2+0.5) * 1e-2
	value := strconv.FormatFloat(f1, 'f', 2, 64)
	fmt.Println(value)
	return value
}
func main(){
	var a float64 = 3

	fmt.Println(FloatToStringTwo(a))
}


/*
		CnArr := []string{"cnc","icdn"}
	AbroadArr := []string{"highwinds","fastly","cdnetworks","cloudflare"}
	GlobalArr := []string{"cnc","icdn","highwinds","fastly","cdnetworks","cloudflare"}
	Providermap := make(map[string][]string)
	Providermap["cn"] = CnArr
	Providermap["abroad"] = AbroadArr
	Providermap["global"] = GlobalArr
	fmt.Println(Providermap["global"][0])
}

/*
data := make(map[string]interface{})
	data = map[string]interface{}{
		"a":"p",
		"b":1,
	}
	fmt.Println(len(data))
}
/*
func QuickSortData(data [][]string, start, end int) ([][]string) {
    if (start < end) {
        base := data[start][0]
	value := data[start]
        left := start
		right := end
        for left < right {
            for left < right && data[right][0] >= base {
                right--
            }
            if left < right {
                data[left] = data[right]
                left++
            }
            for left < right && data[left][0] <= base {
                left++
            }
            if left < right {
                data[right] = data[left]
                right--
            }
        }

        data[left] = value
        QuickSortData(data, start, left - 1)
        QuickSortData(data, left + 1, end)
	}
	return data
}
func StringToInteger64(s string) int64 { //string转int64
	m, _ := strconv.ParseInt(s, 10, 64)
	return m
}

func main(){
	var a [][]string =[][]string{{"1","2","3"},{"0","5","6"},{"89","23","25"},{"12","23","25"}}
	fmt.Println(QuickSortData(a,0,len(a)-1))
}

/*
	datamap := map[int]int{1:8,2:9,5:3,-1:7,3:0}
	var str []int =[]int{}
	for k,v := range datamap{
		str = append(str,k)
		fmt.Println(k,v)
	}
	sort.Ints(str)
	for _, k := range str {
        fmt.Println("Key:", k, "Value:", datamap[k])
    }
}

/*
	func IntegerToString(m int) string{ //int转string
        s := strconv.Itoa(m)
        return s
}
	var domain []string = []string{"llll","kkk","ooo"}
	st := 123
	en := 897
	var str string="&start=" + IntegerToString(st) + "&en="+IntegerToString(en)+""
	for i:=0;i<len(domain);i++ {
		str+="&domain."+IntegerToString(i)+"="+domain[i]
	}
	var s = "dsjadlaskdaskldaskladlkda"+str
	fmt.Println(str)
	fmt.Println(s)
}
/*
	var brr [] map[string]interface{}
	var arr map[string]interface{}
	a := map[string]int{"2015-08-19 08:09:00c":1,"2016-08-19 08:09:00c":2,"2019-08-19 08:09:00e":3,"2020-08-19 08:09:00e":4,"2017-08-19 08:09:00e":6}
	for k,v:=range a {
		for i,j := range a{
			if k[4:] == i[4:]{
				time := i[:20]
				b := i[20:]
				arr["prov"] = b
				arr["list"] = make([] map[string]int,0)
				value := make(map[string]int)
				value[time]=j
				arr["list"] = append(arr["list"],value)
				brr = append(brr,arr)
				delete(a, k)
				fmt.Println(a)
			}

		}
	}
	fmt.Println(arr)
}
/*
	s := "2020-08-19 23:55:00hunan"
	fmt.Println(len(s))
	a := "2020-08-17 23:55:00"
	fmt.Println(len(a))
}
/*
	var Screen []map[string]string
	Screen =append(Screen,map[string]string{"2355":"1","date":"2020-07-21","2350":"4","2345":"0"})
	Screen =append(Screen,map[string]string{"2355":"3","date":"2020-07-13"})
	Screen =append(Screen,map[string]string{"2350":"2","date":"2020-08-19"})
        Screen =append(Screen,map[string]string{"2355":"3","date":"2020-07-13"})
        Screen =append(Screen,map[string]string{"2350":"2","date":"2020-08-19"})
        Screen =append(Screen,map[string]string{"2355":"1","date":"2020-07-21"})
	Screen =append(Screen,map[string]string{"2355":"1","date":"2020-07-21","2350":"6","2345":"0"})
	nummap := make(map[string]int)
	for i:=0; i<len(Screen); i++ {
		for k,v:=range Screen[i]{
			if k != "date" {
				a := fmt.Sprintf("%s %s%s%s%s%s", Screen[i]["date"],  k[0:2], ":", k[2:], ":", "00")
				nummap[a] += StringToInteger(v)
			}
		}
	}
	fmt.Println(nummap)
}
func StringToInteger(s string) int {
	m, _ := strconv.Atoi(s)
	return m
}

/*
func TimeToUnix(s string) (int64){  //时间转时间戳
        loc, _ := time.LoadLocation("Local")
        times, _ := time.ParseInLocation("2006-01-02 15:04:05",s, loc)
        timeunix := times.Unix()
        return timeunix
}
/*
func main(){
	var a []map[string]int
	b := make(map[string]int)
	a =append(a, map[string]int{"s":2,"f":8,"r":1})
	a =append(a, map[string]int{"s":3,"f":8,"r":1})
	a =append(a, map[string]int{"s":2,"f":8,"r":6})
	for i:=0; i<len(a);i++ {
		for k:=range a[i]{
			b[k] += a[i][k]
		}
	}
	fmt.Println(b)
}
func main(){
	var Screen []map[string]string
	 Screen =append(Screen,map[string]string{"f":"456","date":"2020-08-19"})
        Screen =append(Screen,map[string]string{"ll":"89","date":"2020-07-13"})
        Screen =append(Screen,map[string]string{"oo":"56","date":"2020-08-21"})
        Screen =append(Screen,map[string]string{"po":"l6","date":"2020-07-21"})
	fmt.Println(Screen)
	a := quickSort(Screen,0,len(Screen)-1)
	fmt.Println(a)
}
func quickSort(Screen []map[string]string,start int,end int) ([] map[string]string){
	if (end > start){
		m, n := start, end
		value := Screen[0]
		base := TimeToUnix(Screen[0]["date"])
		for {
			if (m < n){
				for{
					if m < n && TimeToUnix(Screen[n]["date"]) >= base {
						n--
					}else{
						Screen[m] = Screen[n]
						break
					}
				}
				for{
					if m < n && TimeToUnix(Screen[m]["date"]) <= base {
						m++
					}else{
						Screen[n] = Screen[m]
						break
					}
				}
			}else{
				break
			}
			Screen[m] = value
			quickSort(Screen, 0, m-1)
			quickSort(Screen, n+1, end)
		}
	}
	return Screen
}
/*
	a := 1
	b := 5555
	c := fmt.Sprintf("%d%d",a,b)
	fmt.Println(c)
	fmt.Println(reflect.TypeOf(c))
}

/*	datamap := map[string]int{"s":1,"k":2,"p":3}
	for k,v := range datamap{
		if k=="s" {
			fmt.Println(datamap["s"])
			fmt.Println(v)

		}
	//	if v==3 {
	//	       	fmt.Println(datamap["p"])
	//	}
	}
}
	var m int64=1596782220 //时间戳转日期
	a := time.Unix(m,0).Format("2006-01-02")
	fmt.Println(a)
}
	var s string = "2020-08-07 1437"
	len := len(s)
	a := s[len-4:len-2]
	c := s[len-2:]
	b := s[0:10]
	loc, _ := time.LoadLocation("Local")
	times, _ := time.ParseInLocation("2006-01-02",b, loc)
	timeunix := times.Unix()
	hourint, _ := strconv.Atoi(a)
	minuteint, _ := strconv.Atoi(c)
	timeunix += int64(hourint) * 3600
	timeunix += int64(minuteint) * 60
	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(c)
	fmt.Println(timeunix)
}
	type a struct {
        b []string
        c [][]string
	}
	var d []string = []string{"dsj","ds45","dsdslf"}
	var e [][]string = [][]string{{"dskd","dskdks","dsjjds"},{"5465","5656","8989"}}
	fmt.Println(reflect.TypeOf(e))
	fmt.Println(reflect.TypeOf(d))
	f := new(a)
	f.b = d
	f.c = e
	fmt.Println(reflect.TypeOf(f))
	fmt.Println(f.b)
	fmt.Println(f.c)
	fmt.Println(f)
}
	var value string
	var cur string
	for i:=0; i<len(htt_str); i++ {
		if string(htt_str[i])==":" {
			if tmp==0 {
				value = fmt.Sprintf("%s%s%s",string(htt_str[i+1]),string(htt_str[i+2]),string(htt_str[i+3]))
				tmp = 1
			}
			if tmp == 1 {
				cur = fmt.Sprintf("%s%s",string(htt_str[i+2]),string(htt_str[i+3]))
			}
		}
	}
		fmt.Printf("%s\n",value)
	fmt.Printf("%d\n",cur)
}
	//var arr []int=[]int{1,2,1,5,6,2,0}
	var brr []string=[]string{"Sdsd","dsidsi","irir5555"}
	var crr []string=[]string{"2222","dsa"}
	nmmap := make(map[string] []string)
//	nummap["arr"]=arr
	nummap["brr"]=brr
	nummap["crr"]=crr
	for k,v := range nummap {
		fmt.Println(k,v)
		fmt.Println(nummap[k][0])
		if k=="brr"{
		for i:=0 ;i<len(nummap[k]); i++ {
			fmt.Printf("%s\n",nummap[k][i])
		}
		}
	}
	var a [][]int =[][]int{{1,2,3},{2,5,6},{89,23,25}}
	var b int
	var c int
	for k,v := range a {
                fmt.Println(k,v)
		b=k
		c=len(v)
	}
	fmt.Println(b)
	fmt.Println(c)
}
*/
