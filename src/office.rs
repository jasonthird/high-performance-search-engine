//! Native Office text extraction. No macros, formulas or external links are executed.
use crate::repo::{Chunk, MAX_CHUNK_BYTES, MAX_CHUNK_LINES};
use anyhow::{bail, Context, Result};
use calamine::Reader as _;
use quick_xml::{events::Event, Reader};
use std::{
    io::{Cursor, Read},
    path::Path,
};

pub const MAX_BYTES: u64 = 64 << 20;
const MAX_EXPANDED: u64 = 128 << 20;
const MAX_TEXT: usize = 32 << 20;

pub fn is_office(path: &str) -> bool {
    path.rsplit_once('.').is_some_and(|(_, ext)| {
        ["docx", "xlsx", "xlsm", "xls", "xlsb", "pptx"]
            .iter()
            .any(|x| ext.eq_ignore_ascii_case(x))
    })
}

pub struct Document {
    parts: Vec<(String, String)>,
}
impl Document {
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
    pub fn chunks(&self, path: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut offset = 0;
        for (name, text) in &self.parts {
            let lines: Vec<_> = text.lines().collect();
            let mut start = 0;
            while start < lines.len() {
                let mut end = start;
                let mut bytes = 0;
                while end < lines.len() && end - start < MAX_CHUNK_LINES {
                    let next = lines[end].len() + usize::from(end > start);
                    if bytes + next > MAX_CHUNK_BYTES {
                        break;
                    }
                    bytes += next;
                    end += 1;
                }
                chunks.push(Chunk {
                    path: path.into(),
                    start_line: offset + start + 1,
                    end_line: offset + end,
                    name: Some(name.clone()),
                    kind: Some("document".into()),
                    parent: None,
                    body: lines[start..end].join("\n"),
                });
                start = end;
            }
            offset += lines.len();
        }
        chunks
    }
}

pub fn extract(path: &Path, bytes: &[u8]) -> Option<Document> {
    match std::panic::catch_unwind(|| extract_inner(path, bytes)) {
        Ok(Ok(doc)) => Some(doc),
        Ok(Err(e)) => {
            eprintln!(
                "warning: skipping Office document {}: {e:#}",
                path.display()
            );
            None
        }
        Err(_) => {
            eprintln!(
                "warning: skipping Office document {}: parser panicked",
                path.display()
            );
            None
        }
    }
}

fn extract_inner(path: &Path, bytes: &[u8]) -> Result<Document> {
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_BYTES,
        "document exceeds size limit"
    );
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut parts = Vec::new();
    // Check compressed containers before handing them to either parser.
    if ext != "xls" {
        archive(bytes)?;
    }
    if ["xlsx", "xlsm", "xls", "xlsb"].contains(&ext.as_str()) {
        let mut book = calamine::open_workbook_auto_from_rs(Cursor::new(bytes))?;
        for name in book.sheet_names().to_vec() {
            let mut text = String::new();
            if let calamine::Sheets::Xlsx(ref mut workbook) = book {
                let mut cells = workbook.worksheet_cells_reader(&name)?;
                while let Some(cell) = cells.next_cell()? {
                    if matches!(cell.get_value(), calamine::DataRef::Empty) {
                        continue;
                    }
                    let (row, col) = cell.get_position();
                    text.push_str(&format!(
                        "{}{}: {}\n",
                        column(col as usize),
                        row + 1,
                        calamine::Data::from(cell.get_value().clone())
                    ));
                    anyhow::ensure!(text.len() <= MAX_TEXT, "worksheet text exceeds limit");
                }
            } else {
                let range = book.worksheet_range(&name)?;
                let (row_offset, col_offset) = range.start().unwrap_or((0, 0));
                for (row, col, value) in range.used_cells() {
                    text.push_str(&format!(
                        "{}{}: {}\n",
                        column(col + col_offset as usize),
                        row + row_offset as usize + 1,
                        value
                    ));
                    anyhow::ensure!(text.len() <= MAX_TEXT, "worksheet text exceeds limit");
                }
            }
            parts.push((format!("sheet {name}"), text));
        }
    } else {
        let mut zip = archive(bytes)?;
        if ext == "docx" {
            parts.push((
                "document".into(),
                xml_text(&xml(&mut zip, "word/document.xml")?)?,
            ));
            if let Some(rels) = optional_xml(&mut zip, "word/_rels/document.xml.rels")? {
                for (_, target, kind) in relationships(&rels)? {
                    if ["header", "footer", "footnotes", "endnotes"]
                        .iter()
                        .any(|k| kind.ends_with(&format!("/{k}")))
                    {
                        let target = resolve("word/document.xml", &target)?;
                        parts.push((target.clone(), xml_text(&xml(&mut zip, &target)?)?));
                    }
                }
            }
        } else if ext == "pptx" {
            let presentation = xml(&mut zip, "ppt/presentation.xml")?;
            let rels = relationships(&xml(&mut zip, "ppt/_rels/presentation.xml.rels")?)?;
            let mut reader = Reader::from_str(&presentation);
            let mut number = 0;
            loop {
                match reader.read_event()? {
                    Event::Start(e) | Event::Empty(e) if e.local_name().as_ref() == b"sldId" => {
                        let mut id = None;
                        for attr in e.attributes() {
                            let a = attr?;
                            if a.key.as_ref().ends_with(b":id") {
                                id = Some(
                                    a.decoded_and_normalized_value(
                                        quick_xml::XmlVersion::Implicit1_0,
                                        reader.decoder(),
                                    )?
                                    .into_owned(),
                                );
                            }
                        }
                        let (_, target, _) = rels
                            .iter()
                            .find(|(key, _, kind)| {
                                Some(key) == id.as_ref() && kind.ends_with("/slide")
                            })
                            .context("missing slide relationship")?;
                        let target = resolve("ppt/presentation.xml", target)?;
                        number += 1;
                        parts.push((
                            format!("slide {number}"),
                            xml_text(&xml(&mut zip, &target)?)?,
                        ));
                        let (dir, file) = target.rsplit_once('/').context("invalid slide path")?;
                        if let Some(notes) =
                            optional_xml(&mut zip, &format!("{dir}/_rels/{file}.rels"))?
                        {
                            for (_, note, kind) in relationships(&notes)? {
                                if kind.ends_with("/notesSlide") {
                                    parts.push((
                                        format!("slide {number} notes"),
                                        xml_text(&xml(&mut zip, &resolve(&target, &note)?)?)?,
                                    ));
                                }
                            }
                        }
                    }
                    Event::DocType(_) => bail!("DTD is unsupported"),
                    Event::Eof => break,
                    _ => {}
                }
            }
        } else {
            bail!("unsupported Office extension");
        }
    }
    anyhow::ensure!(
        parts.iter().map(|(_, s)| s.len()).sum::<usize>() <= MAX_TEXT,
        "extracted text exceeds limit"
    );
    // Bound individual lines without discarding long paragraphs or spreadsheet cells.
    for (_, text) in &mut parts {
        let mut lines = Vec::new();
        for line in text.lines() {
            let mut rest = line.trim_end();
            while rest.len() > 1024 {
                let mut cut = 1024;
                while !rest.is_char_boundary(cut) {
                    cut -= 1;
                }
                lines.push(rest[..cut].to_owned());
                rest = &rest[cut..];
            }
            lines.push(rest.to_owned());
        }
        *text = lines.join("\n").trim().to_owned();
    }
    parts.retain(|(_, text)| !text.is_empty());
    Ok(Document { parts })
}

fn column(mut n: usize) -> String {
    let mut chars = Vec::new();
    loop {
        chars.push((b'A' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    chars.into_iter().rev().collect()
}
type Archive<'a> = zip::ZipArchive<Cursor<&'a [u8]>>;
fn archive(bytes: &[u8]) -> Result<Archive<'_>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
    anyhow::ensure!(archive.len() <= 10_000, "too many archive members");
    let mut total = 0u64;
    for i in 0..archive.len() {
        total = total
            .checked_add(archive.by_index(i)?.size())
            .context("archive size overflow")?;
        anyhow::ensure!(total <= MAX_EXPANDED, "expanded archive exceeds limit");
    }
    Ok(archive)
}
fn optional_xml(zip: &mut Archive<'_>, path: &str) -> Result<Option<String>> {
    match zip.by_name(path) {
        Ok(file) => {
            let mut text = String::new();
            file.take(MAX_TEXT as u64 + 1).read_to_string(&mut text)?;
            anyhow::ensure!(text.len() <= MAX_TEXT, "XML part exceeds limit");
            Ok(Some(text))
        }
        Err(zip::result::ZipError::FileNotFound) => Ok(None),
        Err(e) => Err(e.into()),
    }
}
fn xml(zip: &mut Archive<'_>, path: &str) -> Result<String> {
    optional_xml(zip, path)?.with_context(|| format!("missing {path}"))
}
fn resolve(base: &str, target: &str) -> Result<String> {
    let mut components: Vec<_> = if target.starts_with('/') {
        Vec::new()
    } else {
        base.rsplit_once('/')
            .map(|(dir, _)| dir.split('/').collect())
            .unwrap_or_default()
    };
    for part in target.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                anyhow::ensure!(components.pop().is_some(), "invalid relationship target");
            }
            _ => components.push(part),
        }
    }
    Ok(components.join("/"))
}
fn relationships(text: &str) -> Result<Vec<(String, String, String)>> {
    let mut reader = Reader::from_str(text);
    let mut out = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Start(e) | Event::Empty(e) if e.local_name().as_ref() == b"Relationship" => {
                let (mut id, mut target, mut kind, mut external) = (None, None, None, false);
                for attr in e.attributes() {
                    let a = attr?;
                    let value = a
                        .decoded_and_normalized_value(
                            quick_xml::XmlVersion::Implicit1_0,
                            reader.decoder(),
                        )?
                        .into_owned();
                    match a.key.as_ref() {
                        b"Id" => id = Some(value),
                        b"Target" => target = Some(value),
                        b"Type" => kind = Some(value),
                        b"TargetMode" => external = value == "External",
                        _ => {}
                    }
                }
                if !external {
                    out.push((
                        id.context("missing relationship Id")?,
                        target.context("missing relationship Target")?,
                        kind.context("missing relationship Type")?,
                    ));
                }
            }
            Event::DocType(_) => bail!("DTD is unsupported"),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}
fn xml_text(text: &str) -> Result<String> {
    let mut reader = Reader::from_str(text);
    let mut out = String::new();
    let mut in_text = false;
    loop {
        match reader.read_event()? {
            Event::Start(e) if e.local_name().as_ref() == b"t" => in_text = true,
            Event::End(e) => match e.local_name().as_ref() {
                b"t" => in_text = false,
                b"p" | b"tr" => out.push('\n'),
                b"tc" => out.push('\t'),
                _ => {}
            },
            Event::Empty(e) | Event::Start(e) => match e.local_name().as_ref() {
                b"tab" => out.push('\t'),
                b"br" | b"cr" => out.push('\n'),
                _ => {}
            },
            Event::Text(e) if in_text => out.push_str(&e.decode()?),
            Event::CData(e) if in_text => out.push_str(&e.decode()?),
            Event::GeneralRef(e) if in_text => {
                if let Some(c) = e.resolve_char_ref()? {
                    out.push(c);
                } else {
                    let name = e.decode()?;
                    out.push_str(match name.as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => bail!("unknown XML entity"),
                    });
                }
            }
            Event::DocType(_) => bail!("DTD is unsupported"),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn package(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, text) in parts {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(text.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }
    fn assert_searchable(path: &str, bytes: &[u8], query: &str) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        let index_dir = temp.path().join("index");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(path), bytes).unwrap();
        let indexer = crate::codeindex::RepoIndexer::new(
            &root,
            &index_dir,
            crate::codeindex::BuildOpts {
                embed: false,
                segmented: true,
                quiet: true,
                ..Default::default()
            },
        )
        .unwrap();
        indexer.build().unwrap();
        {
            let index = crate::searcher::AnyIndex::open(&index_dir).unwrap();
            let hits = index.search(query, 5);
            assert!(!hits.results.is_empty(), "{path}: {query}");
            assert!(hits.results[0].id.starts_with(path));
            assert!(crate::repo::snippet_for(&root, &hits.results[0].id, 200)
                .unwrap()
                .contains(query));
        }
        if path.ends_with("docx") {
            std::fs::write(
                root.join(path),
                package(&[("word/document.xml", "<p><t>updateddocument</t></p>")]),
            )
            .unwrap();
            indexer.build().unwrap();
            let index = crate::searcher::AnyIndex::open(&index_dir).unwrap();
            assert!(index.search(query, 5).results.is_empty());
            assert_eq!(index.search("updateddocument", 5).results.len(), 1);
        }
        std::fs::remove_file(root.join(path)).unwrap();
        indexer.build().unwrap();
        let index = crate::searcher::AnyIndex::open(&index_dir).unwrap();
        assert!(index.search(query, 5).results.is_empty());
    }

    #[test]
    fn word_walk_chunks_snippets_and_long_text() {
        let dir = tempfile::tempdir().unwrap();
        let long = "longparagraphλ".repeat(2000);
        let xml = format!("<w:document xmlns:w='urn:word'><w:body><w:p><w:r><w:t>atomic</w:t></w:r><w:r><w:t>commit &amp; rollback &#65;</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>tablevalue</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>{long}</w:t></w:r></w:p></w:body></w:document>");
        std::fs::write(
            dir.path().join("design.DOCX"),
            package(&[("word/document.xml", &xml)]),
        )
        .unwrap();
        std::fs::write(dir.path().join("broken.pptx"), b"not a zip").unwrap();
        let chunks = crate::repo::collect_chunks(dir.path()).unwrap();
        assert!(chunks.len() > 2);
        assert!(chunks[0].body.contains("atomiccommit & rollback A"));
        assert!(chunks[0].body.contains("tablevalue"));
        for chunk in &chunks {
            assert!(chunk.body.len() <= MAX_CHUNK_BYTES);
            assert_eq!(
                crate::repo::snippet_for(dir.path(), &chunk.id(), MAX_CHUNK_LINES).as_deref(),
                Some(chunk.body.as_str())
            );
        }
        let bodies = chunks
            .iter()
            .map(|c| c.body.as_str())
            .collect::<Vec<_>>()
            .join("")
            .replace('\n', "");
        assert!(bodies.ends_with(&long));
        assert_eq!(crate::repo::max_file_bytes("design.DOCX"), MAX_BYTES);
        std::fs::write(
            dir.path().join("design.DOCX"),
            package(&[("word/document.xml", "<p><t>replacementterm</t></p>")]),
        )
        .unwrap();
        let updated = crate::repo::collect_chunks(dir.path()).unwrap();
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].body, "replacementterm");
        assert_searchable(
            "design.docx",
            &std::fs::read(dir.path().join("design.DOCX")).unwrap(),
            "replacementterm",
        );
    }
    #[test]
    fn presentation_order_and_speaker_notes() {
        let bytes = package(&[
            ("ppt/presentation.xml", "<p:presentation xmlns:p='urn:p' xmlns:r='urn:r'><p:sldIdLst><p:sldId id='300' r:id='r2'/><p:sldId id='301' r:id='r1'/></p:sldIdLst></p:presentation>"),
            ("ppt/_rels/presentation.xml.rels", "<Relationships><Relationship Id='r1' Target='slides/slide1.xml' Type='urn:/slide'/><Relationship Id='r2' Target='slides/slide2.xml' Type='urn:/slide'/></Relationships>"),
            ("ppt/slides/slide1.xml", "<p><t>secondslide</t></p>"),
            ("ppt/slides/slide2.xml", "<p><t>firstslide</t></p>"),
            ("ppt/slides/_rels/slide2.xml.rels", "<Relationships><Relationship Id='n1' Target='../notesSlides/notesSlide1.xml' Type='urn:/notesSlide'/></Relationships>"),
            ("ppt/notesSlides/notesSlide1.xml", "<p><t>speakernote</t></p>"),
        ]);
        assert_searchable("deck.pptx", &bytes, "speakernote");
        let doc = extract_inner(Path::new("deck.pptx"), &bytes).unwrap();
        assert_eq!(doc.text(), "firstslide\nspeakernote\nsecondslide");
        let chunks = doc.chunks("deck.pptx");
        assert_eq!(chunks[0].name.as_deref(), Some("slide 1"));
        assert_eq!(chunks[1].name.as_deref(), Some("slide 1 notes"));
        assert_eq!(chunks[2].start_line, 3);
    }
    #[test]
    fn spreadsheet_sheets_sparse_coordinates_and_cached_values() {
        let bytes = package(&[
            ("[Content_Types].xml", "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/></Types>"),
            ("_rels/.rels", "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/></Relationships>"),
            ("xl/workbook.xml", "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets><sheet name='Budget' sheetId='1' r:id='r1'/><sheet name='Notes' sheetId='2' r:id='r2'/></sheets></workbook>"),
            ("xl/_rels/workbook.xml.rels", "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='r1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet1.xml'/><Relationship Id='r2' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet2.xml'/></Relationships>"),
            ("xl/worksheets/sheet1.xml", "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData><row r='4'><c r='C4' t='inlineStr'><is><t>forecastvalue</t></is></c><c r='D4'><f>2+3</f><v>5</v></c></row><row r='1048576'><c r='XFD1048576' t='inlineStr'><is><t>farcell</t></is></c></row></sheetData></worksheet>"),
            ("xl/worksheets/sheet2.xml", "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData><row r='1'><c r='A1' t='inlineStr'><is><t>reviewnotes</t></is></c></row></sheetData></worksheet>"),
        ]);
        assert_searchable("budget.xlsx", &bytes, "forecastvalue");
        let doc = extract_inner(Path::new("budget.xlsx"), &bytes).unwrap();
        assert_eq!(
            doc.text(),
            "C4: forecastvalue\nD4: 5\nXFD1048576: farcell\nA1: reviewnotes"
        );
        assert_eq!(
            doc.chunks("budget.xlsx")[1].name.as_deref(),
            Some("sheet Notes")
        );
    }
    #[test]
    fn malformed_inputs_are_skipped_and_external_entities_rejected() {
        for ext in ["docx", "pptx", "xlsx", "xls", "xlsb"] {
            assert!(extract(Path::new(&format!("bad.{ext}")), b"garbage").is_none());
        }
        assert!(xml_text(
            "<!DOCTYPE x [<!ENTITY a SYSTEM 'file:///tmp/private'>]><p><t>&a;</t></p>"
        )
        .is_err());
        assert!(xml_text("<p><t>broken</p>").is_err());
        assert!(resolve("ppt/presentation.xml", "../../../outside").is_err());
        assert_eq!(column(26), "AA");
    }
}
