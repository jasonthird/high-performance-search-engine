defmodule Widget do
  defstruct width: 0

  def render(%Widget{width: w}), do: String.duplicate(" ", w)

  def compute_total(items), do: Enum.sum(items)
end
