package demo

type Widget struct{ Width int }

// Render draws the widget.
func (w *Widget) Render() int { return w.Width * 2 }

func computeTotal(items []int) int {
	t := 0
	for _, v := range items {
		t += v
	}
	return t
}
