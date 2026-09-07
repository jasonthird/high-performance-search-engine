module Sample

struct Widget
    width::Int
end

function compute_total(items)
    sum(items)
end

render(w::Widget) = repeat(" ", w.width)

end
