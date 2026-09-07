class Widget {
  int width = 0;
  int render() => width * 2;
}

int computeTotal(List<int> items) => items.fold(0, (a, b) => a + b);
