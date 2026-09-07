const os = require("os");

class Widget {
  render(width) {
    return " ".repeat(width);
  }
}

function computeTotal(items) {
  return items.reduce((a, b) => a + b, 0);
}
