//! Jupyter notebook extraction: code and Markdown cells become searchable
//! text. Outputs (which hold the base64 images that make notebooks large),
//! attachments and metadata are never indexed, and nothing is executed.

use crate::repo::Chunk;
use std::path::Path;

/// Notebooks are mostly outputs, so the text-file cap would reject many
/// real ones; parsing cost is bounded by the JSON size, not by anything
/// the cells do.
pub const MAX_BYTES: u64 = 32 << 20;

pub fn is_notebook(path: &str) -> bool {
    path.rsplit_once('.')
        .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("ipynb"))
}

pub struct Notebook {
    cells: Vec<Cell>,
}

struct Cell {
    /// `code`, `markdown` or `raw`, as the notebook declares it.
    kind: String,
    /// Cell source with the trailing newline removed; never empty.
    source: String,
}

impl Notebook {
    /// The extracted text: cells in order, one blank line between them.
    /// Chunk line numbers refer to this text, as for Office documents.
    pub fn text(&self) -> String {
        self.cells
            .iter()
            .map(|c| c.source.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// One chunk per cell (overlong cells split), named `cell N` and
    /// parented by the most recent Markdown heading, so a hit reads as
    /// `analysis.ipynb::Training loop::cell 7`.
    pub fn chunks(&self, path: &str) -> Vec<Chunk> {
        let text = self.text();
        let lines: Vec<&str> = text.lines().collect();
        let mut chunks = Vec::new();
        let mut at = 0usize;
        let mut heading: Option<String> = None;
        for (i, cell) in self.cells.iter().enumerate() {
            let n = cell.source.lines().count().max(1);
            let end = (at + n).min(lines.len());
            if cell.kind == "markdown" {
                if let Some(h) = cell
                    .source
                    .lines()
                    .map(str::trim)
                    .find(|l| l.starts_with('#'))
                {
                    let title = h.trim_start_matches('#').trim();
                    if !title.is_empty() {
                        heading = Some(title.to_string());
                    }
                }
            }
            crate::repo::push_windowed(
                &mut chunks,
                path,
                &lines,
                at,
                end,
                Some(format!("cell {}", i + 1)),
                Some(cell.kind.clone()),
                heading.clone(),
            );
            // The blank separator line between cells.
            at = end + 1;
        }
        chunks
    }
}

pub fn extract(path: &Path, bytes: &[u8]) -> Option<Notebook> {
    match extract_inner(bytes) {
        Ok(nb) => Some(nb),
        Err(e) => {
            eprintln!("warning: skipping notebook {}: {e:#}", path.display());
            None
        }
    }
}

fn extract_inner(bytes: &[u8]) -> anyhow::Result<Notebook> {
    anyhow::ensure!(bytes.len() as u64 <= MAX_BYTES, "notebook exceeds size limit");
    let root: serde_json::Value = serde_json::from_slice(bytes)?;
    // nbformat 4 has a flat `cells` array; nbformat 3 nests them under
    // worksheets. Either way, only `cell_type` and `source`/`input` matter.
    let mut raw: Vec<&serde_json::Value> = Vec::new();
    if let Some(cells) = root.get("cells").and_then(|c| c.as_array()) {
        raw.extend(cells);
    } else if let Some(sheets) = root.get("worksheets").and_then(|w| w.as_array()) {
        for sheet in sheets {
            if let Some(cells) = sheet.get("cells").and_then(|c| c.as_array()) {
                raw.extend(cells);
            }
        }
    } else {
        anyhow::bail!("no cells found");
    }
    let mut cells = Vec::new();
    for cell in raw {
        let kind = cell
            .get("cell_type")
            .and_then(|k| k.as_str())
            .unwrap_or("raw")
            .to_string();
        let source = cell
            .get("source")
            .or_else(|| cell.get("input"))
            .map(source_text)
            .unwrap_or_default();
        let source = source.trim_end_matches(['\n', '\r']).to_string();
        if source.trim().is_empty() {
            continue;
        }
        cells.push(Cell { kind, source });
    }
    Ok(Notebook { cells })
}

/// `source` is either one string or a list of line strings that keep
/// their own newlines.
fn source_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.as_str())
            .collect::<String>(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NB: &str = r###"{
      "cells": [
        {"cell_type": "markdown", "source": ["# Training loop\n", "Fit the model."]},
        {"cell_type": "code", "source": ["import torch\n", "\n", "def fit(model):\n", "    return model\n"],
         "outputs": [{"data": {"image/png": "iVBORw0KGgo="}}]},
        {"cell_type": "code", "source": ""},
        {"cell_type": "markdown", "source": "## Evaluation"},
        {"cell_type": "code", "source": "score = evaluate(model)"}
      ],
      "nbformat": 4
    }"###;

    #[test]
    fn cells_become_chunks_with_heading_parents() {
        let nb = extract_inner(NB.as_bytes()).unwrap();
        let chunks = nb.chunks("analysis.ipynb");
        assert_eq!(chunks.len(), 4, "the empty cell is dropped");
        assert_eq!(chunks[0].title(), "analysis.ipynb::Training loop::cell 1");
        assert_eq!(chunks[1].name.as_deref(), Some("cell 2"));
        assert_eq!(chunks[1].kind.as_deref(), Some("code"));
        assert_eq!(chunks[1].parent.as_deref(), Some("Training loop"));
        assert!(chunks[1].body.contains("def fit(model):"));
        assert!(!chunks[1].body.contains("iVBORw0KGgo="), "outputs are not indexed");
        assert_eq!(chunks[3].parent.as_deref(), Some("Evaluation"));
        // Line ranges address the extracted text, so a snippet can be re-read.
        let text = nb.text();
        let lines: Vec<&str> = text.lines().collect();
        for c in &chunks {
            assert_eq!(lines[c.start_line - 1..c.end_line].join("\n"), c.body, "{}", c.id());
        }
    }

    #[test]
    fn legacy_worksheets_and_bad_json_are_handled() {
        let v3 = r#"{"worksheets": [{"cells": [{"cell_type": "code", "input": ["x = 1\n"]}]}]}"#;
        let nb = extract_inner(v3.as_bytes()).unwrap();
        assert_eq!(nb.text(), "x = 1");
        assert!(extract_inner(b"{\"metadata\": {}}").is_err());
        assert!(extract_inner(b"not json").is_err());
    }
}
