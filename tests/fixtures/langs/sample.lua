local Widget = {}

function Widget.render(width)
  return string.rep(" ", width)
end

local function compute_total(items)
  local t = 0
  for _, v in ipairs(items) do t = t + v end
  return t
end
