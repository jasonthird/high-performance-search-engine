class Widget(width: Int) {
  def render(): Int = width * 2
}

object Totals {
  def computeTotal(items: Seq[Int]): Int = items.sum
}
