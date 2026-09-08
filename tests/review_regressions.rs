//! End-to-end regressions from the September 2026 review.
use high_performance_search_engine::{
    codeindex::{BuildOpts, RepoIndexer},
    indexer::InputDoc,
    searcher::AnyIndex,
    segments::{SegmentedIndex, SegmentedWriter},
};

#[test]
fn successful_migration_keeps_daemon_state_and_indexes_unchanged_sources() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    let index = temp.path().join("index");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("file.toml"), "permission = true\n").unwrap();
    let mut opts = BuildOpts { embed: false, segmented: false, quiet: true, ..Default::default() };
    RepoIndexer::new(&root, &index, opts.clone()).unwrap().build().unwrap();
    let old = AnyIndex::open(&index).unwrap();
    std::fs::create_dir(index.join("watch-state")).unwrap();
    std::fs::write(index.join("watch-state/lease"), "active").unwrap();
    opts.segmented = true;
    let migrated = RepoIndexer::new(&root, &index, opts).unwrap().build().unwrap();
    assert_eq!(migrated.num_docs, 1);
    assert!(index.join("manifest.bin").is_file());
    assert!(!index.join("meta.bin").exists());
    assert_eq!(std::fs::read_to_string(index.join("watch-state/lease")).unwrap(), "active");
    assert_eq!(old.search("permission", 5).results.len(), 1);
    assert_eq!(AnyIndex::open(&index).unwrap().search("permission", 5).results.len(), 1);
}

#[cfg(unix)]
#[test]
fn temporarily_unreadable_source_is_retried_without_losing_old_chunks() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    let index = temp.path().join("index");
    std::fs::create_dir(&root).unwrap();
    let file = root.join("file.toml");
    std::fs::write(&file, "original = true\n").unwrap();
    let indexer = RepoIndexer::new(&root, &index,
        BuildOpts { embed: false, quiet: true, ..Default::default() }).unwrap();
    indexer.build().unwrap();
    let inventory = std::fs::read(index.join("repo.json")).unwrap();
    std::fs::write(&file, "replacement_property = true\n").unwrap();
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o000)).unwrap();
    let unreadable = std::fs::read(&file).is_err();
    let result = if unreadable { Some(indexer.build()) } else { None };
    // Restore even if the build unexpectedly succeeds so the fixture cleans up.
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
    if let Some(result) = result {
        assert!(result.is_err());
        assert_eq!(std::fs::read(index.join("repo.json")).unwrap(), inventory);
        assert_eq!(AnyIndex::open(&index).unwrap().search("original", 5).results.len(), 1);
    }
    assert_eq!(indexer.build().unwrap().num_docs, 1);
    assert_eq!(AnyIndex::open(&index).unwrap().search("replacement_property", 5).results.len(), 1);
}

#[test]
fn bm25_bench_accepts_segmented_indexes_with_deletions() {
    let temp = tempfile::tempdir().unwrap();
    let index = temp.path().join("index");
    let mut writer = SegmentedWriter::open_or_create(&index, true, 2).unwrap();
    for id in ["first", "second"] {
        writer.add_documents(&[InputDoc { id: id.into(), title: "search".into(), body: "needle".into() }]).unwrap();
    }
    writer.delete_document("first").unwrap();
    drop(writer);
    assert_eq!(SegmentedIndex::open(&index).unwrap().num_docs_live(), 1);
    let queries = temp.path().join("queries.txt");
    std::fs::write(&queries, "needle\nneedle search\n").unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_hips"))
        .arg("bench").arg("--index").arg(&index).arg("--queries").arg(queries)
        .env("HIPS_NO_LOG", "1").output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("queries:                 2"), "{text}");
    assert!(text.contains("corpus (1 docs)"), "{text}");
}
