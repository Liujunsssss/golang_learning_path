#if 0
#include <iostream>
#include <cstdlib>
using namespace std;

class Point {
    public:
        Point(int a, int b, char s) : num(a), val(b), str(s) {}
        /*Point(const Point& M){
            num = M.a;
            val = M.b;
            str = M.s;
        }*/
        ~Point(){
            std::cout<<"oooooooooooo"<<std::endl;
        }
        void PrintfResp();
    private:
        int num;
        int val;
        char str;
};
void Point :: PrintfResp() {
    std::cout << "num="<< num << std::endl;
    std::cout << "val="<< val << std::endl;
    std::cout << "str="<< str << std::endl;

}
int main() {
    Point s(1,100,'p');
    s.PrintfResp();
    return 0;
}

int LuckDraw(int sign) {
    int flag = 0;
    flag = sign%3;
    //std::cout<<sign<<std::endl;
    return flag;
}

int main() {
    string arr[3] = {"ssss", "pppppppp", "aaaa"};
    unsigned seed;
    seed = time(0);
    srand(seed);
    while (1) {
        std::cout<<arr[LuckDraw(rand())]<<","<<"kkkkkkkkkkkkkk"<<std::endl;
        sleep(1);
    }
    return 0;
}
#endif
#if 0
int max(int a, int b) {
    return (a>b)?a:b;
}
int main(){
    std::cout<<max(3,7)<<std::endl;
    return 0;
}
#endif