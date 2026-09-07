#include <vector>

class Widget {
public:
    int render(int width) { return width * 2; }
};

int computeTotal(const std::vector<int>& items) {
    int t = 0;
    for (int v : items) t += v;
    return t;
}
