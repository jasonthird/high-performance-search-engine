function Compute-Total {
    param([int[]]$Items)
    ($Items | Measure-Object -Sum).Sum
}

class Widget {
    [int]$Width
    [int] Render() { return $this.Width * 2 }
}
