//! Persistent, exact-text query vectors, shared across repositories and indexes.
//! Each entry is independent; indexing never reads or writes this cache.
use crate::embeddings::CODERANK_DIM;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{mpsc, Arc, OnceLock},
};

const SCHEMA: &str = "hips-query-vector-v1";

#[derive(Serialize, Deserialize)]
struct Entry {
    schema: String,
    encoder: String,
    query: String,
    vector: Vec<f32>,
}

pub(crate) struct DiskQueryCache {
    dir: PathBuf,
    encoder: String,
    writer: Arc<BackgroundWriter>,
}

enum Work {
    Save(PathBuf, Entry),
    Flush(mpsc::SyncSender<()>),
}

struct BackgroundWriter {
    sender: mpsc::SyncSender<Work>,
}

static WRITER: OnceLock<Result<Arc<BackgroundWriter>, String>> = OnceLock::new();

impl BackgroundWriter {
    fn start(
        capacity: usize,
        mut save: impl FnMut(&Path, &Entry) -> anyhow::Result<()> + Send + 'static,
    ) -> std::io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        std::thread::Builder::new()
            .name("query-cache-writer".into())
            .spawn(move || {
                while let Ok(work) = receiver.recv() {
                    match work {
                        Work::Save(dir, entry) => {
                            if let Err(e) = save(&dir, &entry) {
                                if crate::verbosity::verbose() {
                                    eprintln!("note: query cache write failed ({e:#})");
                                }
                            }
                        }
                        Work::Flush(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })?;
        Ok(Self { sender })
    }

    fn enqueue(&self, dir: PathBuf, entry: Entry) -> bool {
        // Cache persistence must never apply backpressure to search responses.
        self.sender.try_send(Work::Save(dir, entry)).is_ok()
    }

    fn flush(&self) {
        let (done, receiver) = mpsc::sync_channel(1);
        if self.sender.send(Work::Flush(done)).is_ok() {
            let _ = receiver.recv();
        }
    }
}

/// Only for shutdown or explicit durability checks, never the response path.
pub(crate) fn flush_pending() {
    if let Some(Ok(writer)) = WRITER.get() {
        writer.flush();
    }
}

pub(crate) fn model_digest(path: &Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

impl DiskQueryCache {
    pub(crate) fn new(root: &Path, recipe: &serde_json::Value) -> anyhow::Result<Self> {
        let bytes = serde_json::to_vec(recipe)?;
        let encoder = format!("{:x}", Sha256::digest(bytes));
        Ok(Self {
            dir: root.join(&encoder),
            encoder,
            writer: WRITER
                .get_or_init(|| {
                    BackgroundWriter::start(64, write_entry)
                        .map(Arc::new)
                        .map_err(|e| e.to_string())
                })
                .as_ref()
                .map_err(|e| anyhow::anyhow!("start query cache writer: {e}"))?
                .clone(),
        })
    }

    pub(crate) fn directory(&self) -> &Path {
        &self.dir
    }

    fn path(&self, query: &str) -> PathBuf {
        self.dir
            .join(format!("{:x}.json", Sha256::digest(query.as_bytes())))
    }

    pub(crate) fn get(&self, query: &str) -> Option<Vec<f32>> {
        // A corrupt or unreadable entry is a miss; inference repairs it.
        let file = std::fs::File::open(self.path(query)).ok()?;
        if file.metadata().ok()?.len() > query.len() as u64 * 6 + 64 * 1024 {
            return None;
        }
        let entry: Entry = serde_json::from_reader(BufReader::new(file)).ok()?;
        if entry.schema != SCHEMA
            || entry.encoder != self.encoder
            || entry.query != query
            || !valid_vector(&entry.vector)
        {
            return None;
        }
        Some(entry.vector)
    }

    pub(crate) fn enqueue(&self, query: &str, vector: &[f32]) -> anyhow::Result<bool> {
        anyhow::ensure!(valid_vector(vector), "invalid query-cache vector");
        let entry = Entry {
            schema: SCHEMA.into(),
            encoder: self.encoder.clone(),
            query: query.into(),
            vector: vector.to_vec(),
        };
        Ok(self.writer.enqueue(self.dir.clone(), entry))
    }

    #[cfg(test)]
    fn put(&self, query: &str, vector: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(valid_vector(vector), "invalid query-cache vector");
        write_entry(
            &self.dir,
            &Entry {
                schema: SCHEMA.into(),
                encoder: self.encoder.clone(),
                query: query.into(),
                vector: vector.to_vec(),
            },
        )
    }
}

fn write_entry(dir: &Path, entry: &Entry) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    // Concurrent processes can replace the same entry atomically. Readers
    // see a complete validated entry; no shared JSON table can lose updates.
    let mut file = tempfile::NamedTempFile::new_in(dir)?;
    {
        let mut writer = BufWriter::new(file.as_file_mut());
        serde_json::to_writer(&mut writer, &entry)?;
        writer.flush()?;
    }
    file.as_file_mut().sync_all()?;
    file.persist(dir.join(format!("{:x}.json", Sha256::digest(entry.query.as_bytes()))))?;
    Ok(())
}

fn valid_vector(vector: &[f32]) -> bool {
    vector.len() == CODERANK_DIM
        && vector.iter().all(|v| v.is_finite())
        && (vector.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row() -> Vec<f32> {
        let mut v = vec![0.0; CODERANK_DIM];
        v[0] = 1.0;
        v
    }

    #[test]
    fn slow_writer_and_full_queue_do_not_block_callers() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let (entered, started) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let mut first = true;
        let writer = Arc::new(
            BackgroundWriter::start(1, move |dir, entry| {
                if first {
                    first = false;
                    entered.send(())?;
                    resume.recv()?;
                }
                write_entry(dir, entry)
            })
            .unwrap(),
        );
        let mut cache = DiskQueryCache::new(dir.path(), &serde_json::json!({})).unwrap();
        cache.writer = writer.clone();
        let cache = Arc::new(cache);
        assert!(cache.enqueue("first", &row()).unwrap());
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        let (finished, received) = mpsc::channel();
        let producer = {
            let cache = cache.clone();
            std::thread::spawn(move || {
                let queued = cache.enqueue("second", &row()).unwrap();
                let overflow = cache.enqueue("third", &row()).unwrap();
                finished.send((queued, overflow)).unwrap();
            })
        };
        let result = received.recv_timeout(Duration::from_secs(1));
        let unsaved = cache.get("first").is_none();
        // Release the simulated stalled disk even if the nonblocking assertion
        // fails, so a regression cannot leave the test process stuck.
        release.send(()).unwrap();
        producer.join().unwrap();
        writer.flush();
        assert_eq!(result.unwrap(), (true, false));
        assert!(unsaved, "caller finished while its save was still blocked");
        assert_eq!(cache.get("first").unwrap(), row());
        assert_eq!(cache.get("second").unwrap(), row());
        assert!(cache.get("third").is_none());
    }

    #[test]
    fn failed_background_write_does_not_stop_later_saves() {
        let dir = tempfile::tempdir().unwrap();
        let blocked = dir.path().join("file");
        std::fs::write(&blocked, "not a directory").unwrap();
        let bad = DiskQueryCache::new(&blocked, &serde_json::json!({})).unwrap();
        let good = DiskQueryCache::new(dir.path(), &serde_json::json!({})).unwrap();
        assert!(bad.enqueue("query", &row()).unwrap());
        assert!(good.enqueue("query", &row()).unwrap());
        flush_pending();
        assert!(bad.get("query").is_none());
        assert_eq!(good.get("query").unwrap(), row());
    }

    #[test]
    fn exact_text_persists_and_is_independent_of_index() {
        let dir = tempfile::tempdir().unwrap();
        let recipe = serde_json::json!({"model":"test", "prefix":"query", "max_tokens":64});
        let cache = DiskQueryCache::new(dir.path(), &recipe).unwrap();
        cache.put("user", &row()).unwrap();
        // All words can be known without the complete issue being cached.
        assert!(cache.get("user user user user user").is_none());
        assert!(cache.get("User").is_none());
        assert!(cache.get(" user").is_none());
        let issue = "user user user user user";
        cache.put(issue, &row()).unwrap();
        let restarted = DiskQueryCache::new(dir.path(), &recipe).unwrap();
        assert_eq!(restarted.get(issue).unwrap(), row());
        assert!(
            DiskQueryCache::new(dir.path(), &serde_json::json!({"model":"changed"}))
                .unwrap()
                .get(issue)
                .is_none()
        );
    }

    #[test]
    fn corrupt_invalid_or_mismatched_entries_are_misses() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskQueryCache::new(dir.path(), &serde_json::json!({})).unwrap();
        cache.put("query", &row()).unwrap();
        std::fs::write(cache.path("query"), b"partial json").unwrap();
        assert!(cache.get("query").is_none());
        assert!(cache.put("query", &vec![0.0; CODERANK_DIM]).is_err());
        assert!(cache.put("query", &vec![f32::NAN; CODERANK_DIM]).is_err());
        assert!(cache.put("query", &[1.0]).is_err());
        cache.put("query", &row()).unwrap();
        let mut entry: Entry =
            serde_json::from_reader(std::fs::File::open(cache.path("query")).unwrap()).unwrap();
        entry.query = "different query".into();
        serde_json::to_writer(std::fs::File::create(cache.path("query")).unwrap(), &entry).unwrap();
        assert!(cache.get("query").is_none());
        cache.put("query", &row()).unwrap();
        assert_eq!(cache.get("query").unwrap(), row());
    }

    #[test]
    fn concurrent_process_style_writers_do_not_lose_entries() {
        let dir = tempfile::tempdir().unwrap();
        let recipe = serde_json::json!({"model":"same"});
        std::thread::scope(|scope| {
            for i in 0..8 {
                let recipe = &recipe;
                let root = dir.path();
                scope.spawn(move || {
                    let cache = DiskQueryCache::new(root, recipe).unwrap();
                    cache.put(&format!("query {i}"), &row()).unwrap();
                    cache.put("shared", &row()).unwrap();
                });
            }
        });
        let cache = DiskQueryCache::new(dir.path(), &recipe).unwrap();
        for i in 0..8 {
            assert_eq!(cache.get(&format!("query {i}")).unwrap(), row());
        }
        assert_eq!(cache.get("shared").unwrap(), row());
    }
    #[test]
    fn model_fingerprint_tracks_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model");
        std::fs::write(&path, b"one").unwrap();
        let first = model_digest(&path).unwrap();
        std::fs::write(&path, b"two").unwrap();
        assert_ne!(first, model_digest(&path).unwrap());
    }
}
