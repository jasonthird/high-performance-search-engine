<?php
class Widget {
    public function render(int $width): int { return $width * 2; }
}
function computeTotal(array $items): int { return array_sum($items); }
