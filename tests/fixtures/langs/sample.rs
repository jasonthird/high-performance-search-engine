/// A widget.
pub struct Widget { width: u32 }

impl Widget {
    /// Draw it.
    pub fn render(&self) -> u32 { self.width * 2 }
}

pub fn compute_total(items: &[u32]) -> u32 { items.iter().sum() }
