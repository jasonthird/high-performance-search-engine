class Widget(val width: Int) {
    fun render(): Int = width * 2
}

fun computeTotal(items: List<Int>): Int = items.sum()
