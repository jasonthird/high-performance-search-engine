type widget = { width : int }

let compute_total items = List.fold_left ( + ) 0 items

let render w = String.make w.width ' '

module Shapes = struct
  let area w = w.width * 2
end
