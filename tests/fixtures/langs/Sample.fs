module Sample

type Widget = { Width: int }

let computeTotal (items: int list) = List.sum items

let render (w: Widget) = String.replicate w.Width " "
