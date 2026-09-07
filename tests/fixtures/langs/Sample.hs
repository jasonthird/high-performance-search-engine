module Sample where

data Widget = Widget { width :: Int }

class Shape a where
  area :: a -> Int

computeTotal :: [Int] -> Int
computeTotal = sum

render :: Widget -> String
render w = replicate (width w) ' '
