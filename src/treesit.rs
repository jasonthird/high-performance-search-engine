//! Declaration-aware chunking through tree-sitter.
//!
//! The keyword heuristic in [`crate::repo`] finds declarations only in
//! languages whose declarations start with a keyword it knows; C, Java, Go
//! methods, JavaScript classes and most others come out as unnamed slabs.
//! Here each supported language is parsed with its tree-sitter grammar and
//! a small query in `tree-sitters/<language>.scm` says which nodes are
//! definitions (`@definition.<kind>`) and where their name is (`@name`).
//! The queries are seeded from the grammars' own `tags.scm` files where
//! they ship one and hand-written otherwise; the folder is the single
//! place to tune what counts as a unit.
//!
//! Chunking is then generic over languages: every definition's span (plus
//! the comment block above it) is a cut point, and the file is covered by
//! the segments between cut points. A segment that starts a definition is
//! named after it; a segment inside a definition but after a nested child
//! (the tail of a class after its last method) keeps the parent's name; a
//! segment before the first definition (imports, module docs) is unnamed,
//! exactly like the heuristic path. Nested definitions therefore become
//! their own chunks with `parent` set, and the parent's chunk holds its
//! header and whatever is not inside a child.
//!
//! Anything the grammar cannot parse (no grammar for the extension, a parse
//! that yields no definitions) falls back to the heuristic chunker, so the
//! worst case is exactly what hips did before.

use std::collections::HashMap;
use std::sync::{LazyLock, OnceLock};

use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

use crate::repo::{self, Chunk};

/// One supported language: how to build its grammar and its query text.
pub struct Lang {
    pub name: &'static str,
    pub exts: &'static [&'static str],
    language: fn() -> Language,
    query_src: &'static str,
    compiled: OnceLock<Option<Query>>,
}

impl Lang {
    fn query(&self) -> Option<&Query> {
        self.compiled
            .get_or_init(|| match Query::new(&(self.language)(), self.query_src) {
                Ok(q) => Some(q),
                Err(e) => {
                    // A broken query file must not take indexing down; the
                    // language just degrades to the heuristic chunker.
                    eprintln!("warning: tree-sitters/{}.scm: {e}", self.name);
                    None
                }
            })
            .as_ref()
    }
}

macro_rules! lang {
    ($name:literal, [$($ext:literal),*], $lang:expr) => {
        lang!($name, [$($ext),*], $lang, $name)
    };
    ($name:literal, [$($ext:literal),*], $lang:expr, $query:literal) => {
        Lang {
            name: $name,
            exts: &[$($ext),*],
            language: || $lang.into(),
            query_src: include_str!(concat!("../tree-sitters/", $query, ".scm")),
            compiled: OnceLock::new(),
        }
    };
}

/// Every grammar compiled into this binary. Extensions are matched
/// case-insensitively; `.m` is sniffed between Objective-C and MATLAB.
pub static LANGUAGES: LazyLock<Vec<Lang>> = LazyLock::new(|| vec![
    lang!("python", ["py", "pyi"], tree_sitter_python::LANGUAGE),
    lang!("javascript", ["js", "mjs", "cjs", "jsx"], tree_sitter_javascript::LANGUAGE),
    lang!("typescript", ["ts", "mts", "cts"], tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
    lang!("tsx", ["tsx"], tree_sitter_typescript::LANGUAGE_TSX, "typescript"),
    lang!("java", ["java"], tree_sitter_java::LANGUAGE),
    lang!("c", ["c"], tree_sitter_c::LANGUAGE),
    lang!("cpp", ["cpp", "cc", "cxx", "hpp", "hh", "hxx", "h", "ipp", "cu", "cuh"], tree_sitter_cpp::LANGUAGE),
    lang!("csharp", ["cs"], tree_sitter_c_sharp::LANGUAGE),
    lang!("go", ["go"], tree_sitter_go::LANGUAGE),
    lang!("rust", ["rs"], tree_sitter_rust::LANGUAGE),
    lang!("php", ["php", "phtml"], tree_sitter_php::LANGUAGE_PHP),
    lang!("ruby", ["rb", "rake", "gemspec"], tree_sitter_ruby::LANGUAGE),
    lang!("swift", ["swift"], tree_sitter_swift::LANGUAGE),
    lang!("kotlin", ["kt", "kts"], tree_sitter_kotlin_ng::LANGUAGE),
    lang!("scala", ["scala", "sc"], tree_sitter_scala::LANGUAGE),
    lang!("dart", ["dart"], tree_sitter_dart::LANGUAGE),
    lang!("lua", ["lua"], tree_sitter_lua::LANGUAGE),
    lang!("perl", ["pl", "pm", "t"], tree_sitter_perl::LANGUAGE),
    lang!("r", ["r"], tree_sitter_r::LANGUAGE),
    lang!("objc", ["mm"], tree_sitter_objc::LANGUAGE),
    lang!("matlab", [], tree_sitter_matlab::LANGUAGE),
    lang!("bash", ["sh", "bash", "zsh"], tree_sitter_bash::LANGUAGE),
    lang!("powershell", ["ps1", "psm1", "psd1"], tree_sitter_powershell::LANGUAGE),
    lang!("sql", ["sql"], tree_sitter_sequel::LANGUAGE),
    lang!("haskell", ["hs"], tree_sitter_haskell::LANGUAGE),
    lang!("elixir", ["ex", "exs"], tree_sitter_elixir::LANGUAGE),
    lang!("erlang", ["erl", "hrl"], tree_sitter_erlang::LANGUAGE),
    lang!("ocaml", ["ml"], tree_sitter_ocaml::LANGUAGE_OCAML),
    lang!("julia", ["jl"], tree_sitter_julia::LANGUAGE),
    lang!("zig", ["zig"], tree_sitter_zig::LANGUAGE),
    lang!("groovy", ["groovy", "gradle", "gvy"], tree_sitter_groovy::LANGUAGE),
    lang!("fortran", ["f", "f90", "f95", "f03", "f08", "for"], tree_sitter_fortran::LANGUAGE),
    lang!("pascal", ["pas", "pp", "dpr"], tree_sitter_pascal::LANGUAGE),
    lang!("ada", ["adb", "ads"], tree_sitter_ada::LANGUAGE),
    lang!("solidity", ["sol"], tree_sitter_solidity::LANGUAGE),
    lang!("hcl", ["tf", "tfvars", "hcl"], tree_sitter_hcl::LANGUAGE),
    lang!("nix", ["nix"], tree_sitter_nix::LANGUAGE),
    lang!("elm", ["elm"], tree_sitter_elm::LANGUAGE),
    lang!("cmake", ["cmake"], tree_sitter_cmake::LANGUAGE),
    lang!("asm", ["s", "asm"], tree_sitter_asm::LANGUAGE),
    lang!("markdown", ["md", "markdown"], tree_sitter_md::LANGUAGE),
]);

/// Extensions that reach a grammar, for the walker's file filter.
pub fn extensions() -> impl Iterator<Item = &'static str> {
    LANGUAGES
        .iter()
        .flat_map(|l| l.exts.iter().copied())
        .chain(std::iter::once("m"))
}

fn by_name(name: &str) -> Option<&'static Lang> {
    LANGUAGES.iter().find(|l| l.name == name)
}

/// The grammar for a file, chosen by extension (and content for `.m`).
pub fn language_for(rel: &str, text: &str) -> Option<&'static Lang> {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    if name == "CMakeLists.txt" {
        return by_name("cmake");
    }
    let ext = name.rsplit_once('.')?.1.to_ascii_lowercase();
    if ext == "m" {
        // Objective-C and MATLAB share `.m`; ObjC files announce themselves.
        let objc = text.contains("@interface")
            || text.contains("@implementation")
            || text.contains("#import")
            || text.contains("@end");
        return by_name(if objc { "objc" } else { "matlab" });
    }
    LANGUAGES.iter().find(|l| l.exts.contains(&ext.as_str()))
}

/// A definition found by the query: 1-based inclusive lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Def {
    pub start_line: usize,
    pub end_line: usize,
    /// First line of the comment block above the definition, when the
    /// query captured one (`@doc`); otherwise `start_line`.
    pub doc_line: usize,
    pub name: String,
    pub kind: String,
    /// The node has no `body` child: a prototype, a typedef, a `#define`,
    /// a one-line declaration. Runs of these merge into one chunk.
    pub bodyless: bool,
}

/// Cap on a merged run of one-line units (see `cover`).
const RUN_LINES: usize = 24;

/// Kinds that are not units of their own: a field or constant lives inside
/// the chunk of whatever encloses it (or the file preamble).
fn is_member_kind(kind: &str) -> bool {
    matches!(
        kind,
        "constant" | "variable" | "property" | "field" | "parameter" | "signature" | "attribute"
    )
}

/// Kinds whose node spans one line but which open a region that runs to
/// the next definition (assembly labels).
fn is_open_ended(kind: &str) -> bool {
    kind == "label"
}

/// Parse `text` and return its definitions, sorted by start then by span
/// (largest first). `None` when the grammar produced nothing usable.
pub fn definitions(lang: &Lang, text: &str) -> Option<Vec<Def>> {
    let query = lang.query()?;
    let mut parser = Parser::new();
    parser.set_language(&(lang.language)()).ok()?;
    let tree = parser.parse(text, None)?;
    let names = query.capture_names();
    let def_idx: Vec<(u32, String)> = names
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            n.strip_prefix("definition.")
                .map(|kind| (i as u32, kind.to_string()))
        })
        .collect();
    let name_idx = names.iter().position(|n| *n == "name")? as u32;
    let doc_idx = names.iter().position(|n| *n == "doc").map(|i| i as u32);

    let bytes = text.as_bytes();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), bytes);
    let mut raw: Vec<(tree_sitter::Node, Def)> = Vec::new();
    while let Some(m) = matches.next() {
        for (idx, kind) in &def_idx {
            for node in m.nodes_for_capture_index(*idx) {
                // Prefer a `@name` inside the definition node; a pattern may
                // bind several alternatives, and a few queries put the name
                // beside the node (Nix bindings), so fall back to any.
                let name_node = m
                    .nodes_for_capture_index(name_idx)
                    .find(|n| n.start_byte() >= node.start_byte() && n.end_byte() <= node.end_byte())
                    .or_else(|| m.nodes_for_capture_index(name_idx).next());
                let name = name_node
                    .and_then(|n| n.utf8_text(bytes).ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let Some(name) = name else { continue };
                let start_line = node.start_position().row + 1;
                let doc_line = doc_idx
                    .and_then(|d| m.nodes_for_capture_index(d).map(|n| n.start_position().row + 1).min())
                    .filter(|l| *l < start_line)
                    .unwrap_or(start_line);
                raw.push((
                    node,
                    Def {
                        start_line,
                        end_line: last_line(&node),
                        doc_line,
                        name: first_line(&name),
                        kind: kind.clone(),
                        bodyless: is_bodyless(&node),
                    },
                ));
            }
        }
    }
    if raw.is_empty() {
        return None;
    }
    // Some grammars expose a header as its own node (Fortran's
    // `function_statement`, a C++ declarator): a single-line capture whose
    // parent starts on the same line is a header, and the unit is the
    // parent — provided the parent holds no other definition, which is
    // what separates a function body from a whole declaration list that
    // merely begins on that line.
    let starts: Vec<usize> = raw.iter().map(|(_, d)| d.start_line).collect();
    let defs: Vec<Def> = raw
        .into_iter()
        .map(|(mut node, mut def)| {
            while def.start_line == last_line(&node) {
                let Some(parent) = node.parent() else { break };
                let (ps, pe) = (parent.start_position().row + 1, last_line(&parent));
                let holds_other = starts.iter().any(|s| *s > def.start_line && *s <= pe);
                if parent.parent().is_none() || ps != def.start_line || pe == def.end_line || holds_other {
                    break;
                }
                node = parent;
            }
            def.end_line = last_line(&node);
            def.bodyless = is_bodyless(&node);
            def
        })
        .collect();
    let mut defs = defs;
    defs.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then((b.end_line - b.start_line).cmp(&(a.end_line - a.start_line)))
    });
    // Two patterns matching the same node (or an outer and inner node on
    // the same line) describe one unit: keep the larger, first-sorted one.
    defs.dedup_by(|b, a| a.start_line == b.start_line);
    Some(defs)
}

/// A definition with no body: a prototype, a typedef, a `#define`, an
/// abstract method. Grammars name the body child differently (`body:`
/// field, `function_body`, `class_body`, `block`, `compound_statement`,
/// `declaration_list`), so look for any of those among the named children.
fn is_bodyless(node: &tree_sitter::Node) -> bool {
    if node.child_by_field_name("body").is_some() {
        return false;
    }
    let mut cursor = node.walk();
    let has_body = node.named_children(&mut cursor).any(|c| {
        let k = c.kind();
        k.contains("body")
            || k.contains("block")
            || k == "compound_statement"
            || k.ends_with("declaration_list")
            || k.ends_with("enumerator_list")
    });
    !has_body
}

/// Kinds whose bodyless one-liners merge into runs: the things headers
/// are made of. Functions join the list only for languages with
/// prototypes (`merge_prototypes`); in expression-bodied languages a
/// one-line function is the whole function and keeps its own chunk.
fn is_mergeable_kind(kind: &str, merge_prototypes: bool) -> bool {
    match kind {
        "macro" | "type" | "constant" | "variable" | "field" => true,
        "function" | "method" => merge_prototypes,
        _ => false,
    }
}

/// Languages whose headers carry bodyless function declarations.
fn has_prototypes(lang: &str) -> bool {
    matches!(lang, "c" | "cpp" | "objc")
}

/// Last line a node occupies, 1-based. Grammars that fold the terminating
/// newline into a statement (Fortran, some Lisp) report an end position
/// at column 0 of the next row; that row is not part of the node.
fn last_line(node: &tree_sitter::Node) -> usize {
    let end = node.end_position();
    if end.column == 0 && end.row > node.start_position().row {
        end.row
    } else {
        end.row + 1
    }
}

/// Multi-line names (a Markdown heading with trailing markup, a Scala
/// pattern) are cut to their first line.
fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

/// Chunk a file using its grammar; `None` means "use the heuristic".
pub fn chunk(rel: &str, text: &str) -> Option<Vec<Chunk>> {
    let lang = language_for(rel, text)?;
    let defs = definitions(lang, text)?;
    Some(cover(rel, text, &defs, has_prototypes(lang.name)))
}

/// Turn definitions into chunks covering the whole file. See the module
/// docs for the rules.
pub fn cover(rel: &str, text: &str, defs: &[Def], merge_prototypes: bool) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    if n == 0 {
        return Vec::new();
    }
    // Units: definitions that own a chunk. Members stay inside their parent.
    let mut units: Vec<Def> = defs.iter().filter(|d| !is_member_kind(&d.kind)).cloned().collect();
    let members: Vec<&Def> = defs.iter().filter(|d| is_member_kind(&d.kind)).collect();
    // Pull each unit's start up over the comment block above it, the way
    // the heuristic chunker does, unless the query already captured a doc.
    // A type signature with the unit's name just above it (Haskell, Elm,
    // OCaml interfaces) belongs to the unit too.
    let mut floor = 1usize;
    for u in units.iter_mut() {
        let mut at = u.doc_line.min(u.start_line);
        if at == u.start_line {
            loop {
                if at > floor && repo::is_doc_or_attr(lines[at - 2]) {
                    at -= 1;
                    continue;
                }
                // A comment block separated by one blank line still belongs
                // to the declaration below it: JSDoc above a blank line, a
                // `// ---- section ----` banner above a documented function.
                if at > floor + 1
                    && lines[at - 2].trim().is_empty()
                    && !lines[at - 1].trim().is_empty()
                    && repo::is_doc_or_attr(lines[at - 3])
                {
                    at -= 1;
                    continue;
                }
                let sig = members.iter().find(|m| {
                    m.kind == "signature" && m.name == u.name && m.end_line + 1 >= at.saturating_sub(1) && m.end_line < at
                });
                match sig {
                    Some(m) if m.start_line >= floor => at = m.start_line,
                    _ => break,
                }
            }
        }
        u.doc_line = at.max(floor);
        floor = u.start_line;
    }
    // Open-ended kinds run to the next unit's start.
    for i in 0..units.len() {
        if is_open_ended(&units[i].kind) {
            let next = units.get(i + 1).map(|d| d.doc_line - 1).unwrap_or(n);
            units[i].end_line = next.max(units[i].end_line);
        }
    }
    // Cut points: every unit's (doc) start and the line after its end.
    let mut cuts: Vec<usize> = vec![1, n + 1];
    for u in &units {
        cuts.push(u.doc_line);
        cuts.push(u.end_line + 1);
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts.retain(|c| *c >= 1 && *c <= n + 1);

    // Segments between cut points, then a merge pass: a segment that is
    // not the start of a unit and holds at most two non-blank lines (the
    // closing brace of a class after its last method, an `end`) joins the
    // segment before it rather than becoming a chunk of its own.
    struct Seg {
        a: usize,
        b: usize,
        name: Option<String>,
        kind: Option<String>,
        parent: Option<String>,
        starts_unit: bool,
        bodyless: bool,
    }
    let mut segs: Vec<Seg> = Vec::new();
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]); // lines a..b-1, 1-based inclusive
        if a >= b {
            continue;
        }
        // The unit starting here, else the innermost unit containing `a`.
        let starting = units.iter().filter(|u| u.doc_line == a).min_by_key(|u| u.end_line - u.doc_line);
        let owner = starting.or_else(|| {
            units
                .iter()
                .filter(|u| u.doc_line <= a && u.end_line >= a)
                .min_by_key(|u| u.end_line - u.doc_line)
        });
        let parent = owner.and_then(|o| {
            units
                .iter()
                .filter(|u| u.doc_line <= o.doc_line && u.end_line >= o.end_line && !std::ptr::eq(*u, o))
                .min_by_key(|u| u.end_line - u.doc_line)
        });
        segs.push(Seg {
            a,
            b,
            name: owner.map(|o| o.name.clone()),
            kind: owner.map(|o| o.kind.clone()),
            parent: parent.map(|p| p.name.clone()),
            starts_unit: starting.is_some(),
            // Mergeable: a bodyless one- or two-line unit of a header-like
            // kind, judged on the unit's own span, not the segment's.
            bodyless: starting.is_some_and(|s| {
                s.bodyless
                    && is_mergeable_kind(&s.kind, merge_prototypes)
                    && s.end_line + 1 - s.doc_line <= 2
            }),
        });
    }
    // Runs of one- and two-line units — prototypes in a C header, a block
    // of `#define`s, typedefs — merge into chunks of up to `RUN_LINES`
    // lines named after the first, so a header does not become a thousand
    // single-line documents. Real (multi-line) definitions stay separate.
    let mut merged: Vec<Seg> = Vec::new();
    let mut run_open = false;
    for seg in segs {
        let non_blank = lines[seg.a - 1..seg.b - 1].iter().filter(|l| !l.trim().is_empty()).count();
        let tiny_unit = seg.starts_unit && seg.bodyless;
        if let Some(prev) = merged.last_mut() {
            let small_tail = !seg.starts_unit && non_blank <= 2;
            let fits = seg.b - prev.a <= repo::MAX_CHUNK_LINES;
            if small_tail && fits {
                prev.b = seg.b;
                continue;
            }
            if tiny_unit && run_open && seg.b - prev.a <= RUN_LINES {
                prev.b = seg.b;
                continue;
            }
        }
        run_open = tiny_unit;
        merged.push(seg);
    }
    let mut chunks = Vec::new();
    for seg in merged {
        repo::push_split_full(
            &mut chunks,
            rel,
            &lines,
            seg.a - 1,
            seg.b - 1,
            seg.name,
            seg.kind,
            seg.parent,
        );
    }
    chunks
}

/// Which grammar and how many definitions a file yields — for `hips chunks`.
pub fn explain(rel: &str, text: &str) -> Option<(&'static str, Vec<Def>)> {
    let lang = language_for(rel, text)?;
    Some((lang.name, definitions(lang, text).unwrap_or_default()))
}

/// The parse tree as an S-expression, for writing queries.
pub fn sexp(rel: &str, text: &str) -> Option<String> {
    let lang = language_for(rel, text)?;
    let mut parser = Parser::new();
    parser.set_language(&(lang.language)()).ok()?;
    Some(parser.parse(text, None)?.root_node().to_sexp())
}

/// Names of all grammars, for `--help` and status output.
pub fn language_names() -> Vec<&'static str> {
    LANGUAGES.iter().map(|l| l.name).collect()
}

#[allow(dead_code)]
fn _kinds_seen(defs: &[Def]) -> HashMap<&str, usize> {
    let mut m = HashMap::new();
    for d in defs {
        *m.entry(d.kind.as_str()).or_insert(0) += 1;
    }
    m
}
