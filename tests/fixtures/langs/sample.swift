struct Widget {
    var width: Int
    func render() -> Int { return width * 2 }
}

func computeTotal(_ items: [Int]) -> Int { items.reduce(0, +) }
