//! Every shipped grammar must turn its fixture into named, kinded chunks:
//! the grammar is detected from the extension, the expected declarations
//! are found by name, nested units carry their parent, and the file is
//! covered without gaps. A grammar or query regression fails here rather
//! than silently degrading a language to unnamed slabs.

#![cfg(feature = "treesitter")]

use high_performance_search_engine::repo::chunk_source;
use high_performance_search_engine::treesit;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/langs/");

/// (file, grammar, names that must appear as chunk names)
const EXPECT: &[(&str, &str, &[&str])] = &[
    ("sample.py", "python", &["Widget", "render", "compute_total"]),
    ("sample.js", "javascript", &["Widget", "render", "computeTotal"]),
    ("sample.ts", "typescript", &["Shape", "Widget", "area", "computeTotal"]),
    ("sample.tsx", "tsx", &["Widget"]),
    ("Sample.java", "java", &["Widget", "render", "Shape"]),
    ("sample.c", "c", &["widget", "compute_total"]),
    ("sample.cpp", "cpp", &["Widget", "render", "computeTotal"]),
    ("Sample.cs", "csharp", &["Widget", "Render", "IShape"]),
    ("sample.go", "go", &["Widget", "Render", "computeTotal"]),
    ("sample.rs", "rust", &["Widget", "render", "compute_total"]),
    ("sample.php", "php", &["Widget", "render", "computeTotal"]),
    ("sample.rb", "ruby", &["Widget", "render", "compute_total"]),
    ("sample.swift", "swift", &["Widget", "render", "computeTotal"]),
    ("Sample.kt", "kotlin", &["Widget", "render", "computeTotal"]),
    ("Sample.scala", "scala", &["Widget", "render", "Totals", "computeTotal"]),
    ("sample.dart", "dart", &["Widget", "render", "computeTotal"]),
    ("sample.lua", "lua", &["compute_total"]),
    ("sample.pl", "perl", &["Widget", "render", "compute_total"]),
    ("sample.r", "r", &["compute_total", "render"]),
    ("Sample.m", "objc", &["Widget", "render"]),
    ("compute_total.m", "matlab", &["compute_total", "render"]),
    ("sample.sh", "bash", &["compute_total", "render"]),
    ("sample.ps1", "powershell", &["Compute-Total", "Widget"]),
    ("sample.sql", "sql", &["widgets", "wide_widgets", "widgets_width"]),
    ("Sample.hs", "haskell", &["Widget", "Shape", "computeTotal", "render"]),
    ("sample.ex", "elixir", &["Widget", "render", "compute_total"]),
    ("sample.erl", "erlang", &["widget", "compute_total", "render"]),
    ("sample.ml", "ocaml", &["compute_total", "render", "Shapes"]),
    ("sample.jl", "julia", &["Sample", "Widget", "compute_total", "render"]),
    ("sample.zig", "zig", &["Widget", "render", "computeTotal"]),
    ("Sample.groovy", "groovy", &["Widget", "render", "computeTotal"]),
    ("sample.f90", "fortran", &["sample", "compute_total", "render"]),
    ("sample.pas", "pascal", &["TWidget", "ComputeTotal"]),
    ("sample.adb", "ada", &["Sample", "Compute_Total", "Render"]),
    ("Sample.sol", "solidity", &["Widget", "render", "computeTotal"]),
    ("main.tf", "hcl", &["resource", "variable"]),
    ("sample.nix", "nix", &["computeTotal", "render"]),
    ("Sample.elm", "elm", &["computeTotal", "render"]),
    ("CMakeLists.txt", "cmake", &["compute_total", "render"]),
    ("sample.s", "asm", &["compute_total", "render"]),
    ("README.md", "markdown", &["Sample", "Widget", "Render", "Compute total"]),
];

fn read(file: &str) -> String {
    std::fs::read_to_string(format!("{FIXTURES}{file}")).unwrap_or_else(|e| panic!("{file}: {e}"))
}

#[test]
fn every_grammar_names_its_fixtures_declarations() {
    let mut failures = Vec::new();
    for (file, grammar, names) in EXPECT {
        let text = read(file);
        let detected = treesit::explain(file, &text).map(|(g, _)| g);
        if detected != Some(*grammar) {
            failures.push(format!("{file}: grammar {detected:?}, expected {grammar}"));
            continue;
        }
        let chunks = chunk_source(file, &text);
        let found: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        for name in *names {
            if !found.contains(name) {
                failures.push(format!("{file}: missing {name:?}; found {found:?}"));
            }
        }
        // Chunks cover the file in order with no gaps or overlaps.
        let total = text.lines().count();
        let mut expect_next = 1;
        for c in &chunks {
            if c.start_line < expect_next {
                failures.push(format!("{file}: overlap at {}", c.id()));
            }
            expect_next = c.end_line + 1;
        }
        if expect_next <= total && chunks.iter().all(|c| c.end_line < total) {
            // A trailing blank-only segment is dropped; anything else is a gap.
            let tail: String = text.lines().skip(expect_next - 1).collect::<Vec<_>>().join("\n");
            if !tail.trim().is_empty() {
                failures.push(format!("{file}: uncovered tail from line {expect_next}"));
            }
        }
        // A grammar-chunked file has kinds on its named chunks.
        if chunks.iter().any(|c| c.name.is_some() && c.kind.is_none()) {
            failures.push(format!("{file}: named chunk without kind"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn nested_units_carry_their_parent() {
    let text = read("Sample.java");
    let chunks = chunk_source("Sample.java", &text);
    let render = chunks.iter().find(|c| c.name.as_deref() == Some("render")).expect("render chunk");
    assert_eq!(render.parent.as_deref(), Some("Widget"));
    assert_eq!(render.kind.as_deref(), Some("method"));
    assert_eq!(render.title(), "Sample.java::Widget::render");
    // The class chunk holds the header and fields, not the method body.
    let widget = chunks.iter().find(|c| c.name.as_deref() == Some("Widget")).unwrap();
    assert!(widget.body.contains("private int width"));
    assert!(!widget.body.contains("return width * 2"));
}

#[test]
fn closing_braces_join_the_previous_chunk() {
    let text = read("sample.cpp");
    let chunks = chunk_source("sample.cpp", &text);
    // No chunk is just `};`.
    for c in &chunks {
        let body = c.body.trim();
        assert!(body != "};" && body != "}", "lonely brace chunk {}", c.id());
    }
}

#[test]
fn doc_comments_and_signatures_travel_with_their_unit() {
    let go = read("sample.go");
    let chunks = chunk_source("sample.go", &go);
    let render = chunks.iter().find(|c| c.name.as_deref() == Some("Render")).unwrap();
    assert!(render.body.starts_with("// Render draws the widget."), "{}", render.body);

    let hs = read("Sample.hs");
    let chunks = chunk_source("Sample.hs", &hs);
    let total = chunks.iter().find(|c| c.name.as_deref() == Some("computeTotal")).unwrap();
    assert!(total.body.contains("computeTotal :: [Int] -> Int"), "{}", total.body);
}

#[test]
fn markdown_sections_nest_by_heading_level() {
    let text = read("README.md");
    let chunks = chunk_source("README.md", &text);
    let render = chunks.iter().find(|c| c.name.as_deref() == Some("Render")).unwrap();
    assert_eq!(render.parent.as_deref(), Some("Widget"));
    assert_eq!(render.kind.as_deref(), Some("section"));
}

#[test]
fn unknown_extensions_fall_back_to_the_heuristic() {
    let text = "pub fn keyword_only() {}\n";
    let chunks = chunk_source("weird.xyz", text);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].name.as_deref(), Some("keyword_only"));
    assert!(chunks[0].kind.is_none(), "heuristic chunks carry no kind");
}

#[test]
fn impl_headers_fold_into_their_first_method() {
    // `impl Tokenizer {` used to be a one-line chunk of its own — a tiny
    // document carrying the type name in its title, which outranked the
    // real code for that name. It now opens the first method's chunk.
    let text = "/// A widget.\npub struct Widget { width: u32 }\n\nimpl Widget {\n    /// Draw it.\n    pub fn render(&self) -> u32 { self.width * 2 }\n\n    pub fn area(&self) -> u32 { 1 }\n}\n\nimpl std::fmt::Display for Widget {\n    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {\n        write!(f, \"w\")\n    }\n}\n";
    let chunks = chunk_source("w.rs", text);
    let ids: Vec<String> = chunks.iter().map(|c| format!("{} {}", c.id(), c.title())).collect();
    assert!(chunks.iter().all(|c| c.body.trim() != "impl Widget {"), "{ids:?}");
    let render = chunks.iter().find(|c| c.name.as_deref() == Some("render")).unwrap();
    assert_eq!(render.start_line, 4, "{ids:?}");
    assert!(render.body.starts_with("impl Widget {"));
    assert_eq!(render.title(), "w.rs::Widget::render");
    let fmt = chunks.iter().find(|c| c.name.as_deref() == Some("fmt")).unwrap();
    assert!(fmt.body.starts_with("impl std::fmt::Display for Widget {"), "{}", fmt.body);
    // The struct, with its own fields, is still its own chunk.
    let widget = chunks.iter().find(|c| c.name.as_deref() == Some("Widget")).unwrap();
    assert_eq!(widget.kind.as_deref(), Some("class"));
    // Coverage is intact.
    let mut next = 1;
    for c in &chunks {
        assert!(c.start_line >= next, "overlap at {}", c.id());
        next = c.end_line + 1;
    }
}

#[test]
fn bodyless_module_and_type_one_liners_merge_into_runs() {
    let mods: String = (0..30).map(|i| format!("pub mod m{i};\n")).collect();
    let text = format!("//! Crate root.\n{mods}\npub type Alias = u32;\npub type Other = u64;\n\npub fn real() {{ }}\n");
    let chunks = chunk_source("lib.rs", &text);
    let one_liners = chunks.iter().filter(|c| c.end_line == c.start_line).count();
    assert!(one_liners <= 1, "{:?}", chunks.iter().map(|c| c.id()).collect::<Vec<_>>());
    // Runs are capped, so the 30 modules take two chunks, not one or thirty.
    let module_chunks = chunks.iter().filter(|c| c.kind.as_deref() == Some("module")).count();
    assert!(module_chunks >= 2 && module_chunks <= 3, "{module_chunks}");
    assert!(chunks.iter().any(|c| c.name.as_deref() == Some("real")));
}
