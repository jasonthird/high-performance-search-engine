module Sample exposing (computeTotal, render)

type alias Widget = { width : Int }

computeTotal : List Int -> Int
computeTotal items =
    List.sum items

render : Widget -> String
render w =
    String.repeat w.width " "
