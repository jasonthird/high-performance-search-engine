//! On-demand, versioned native grammars. Keep the library alive with its language.
use anyhow::{bail, Context, Result};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tree_sitter::{Language, Parser};
use tree_sitter_language::LanguageFn;

// Immutable release: bump whenever a grammar or its build changes.
const RELEASE: &str = "grammars-v1";
const RELEASE_URL: &str =
    "https://github.com/jasonthird/high-performance-search-engine/releases/download";

pub(crate) struct LoadedGrammar {
    pub language: Language,
    // Fields drop in order: the language must be dropped before the library.
    _library: Library,
}

fn platform() -> Result<String> {
    if cfg!(target_env = "musl") {
        bail!("prebuilt grammars require glibc; use HIPS_GRAMMAR_DIR for a native build");
    }
    let target = match (std::env::consts::ARCH, std::env::consts::OS) {
        ("aarch64", "macos") => "aarch64-apple-darwin",
        ("x86_64", "macos") => "x86_64-apple-darwin",
        ("aarch64", "linux") => "aarch64-unknown-linux-gnu",
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu",
        _ => bail!("no grammar artifacts for this platform"),
    };
    Ok(target.into())
}

fn cache_dir(target: &str) -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("set XDG_CACHE_HOME or HOME for the grammar cache")?;
    Ok(base.join("csearch/grammars").join(RELEASE).join(target))
}

pub(crate) fn load(name: &str) -> Result<LoadedGrammar> {
    let filename = format!("libhips_grammar_{name}.{}", std::env::consts::DLL_EXTENSION);
    // Packages may bundle any subset. Installed libraries take precedence and
    // stay in place; only missing languages use the user's cache/download path.
    if let Some(dir) = std::env::var_os("HIPS_GRAMMAR_DIR") {
        let path = PathBuf::from(dir).join(&filename);
        if path.is_file() {
            return open(&path);
        }
    }
    let target = platform()?;
    let path = cache_dir(&target)?.join(&filename);
    if !path.is_file() {
        if std::env::var_os("HIPS_GRAMMAR_OFFLINE").as_deref() == Some(std::ffi::OsStr::new("1")) {
            bail!("{name} is neither installed nor cached and HIPS_GRAMMAR_OFFLINE=1");
        }
        let url = format!("{RELEASE_URL}/{RELEASE}/{target}-{filename}");
        eprintln!("Downloading {name} grammar …");
        download(&url, &path)?;
    }
    open(&path)
}

fn open(path: &Path) -> Result<LoadedGrammar> {
    // SAFETY: only our versioned release artifacts (or an explicitly trusted
    // local directory) enter here. They export a generated tree-sitter language.
    // The owning library remains alive for every parser/query using the language.
    let library =
        unsafe { Library::new(path) }.with_context(|| format!("loading {}", path.display()))?;
    let language = unsafe {
        let entry = library.get::<unsafe extern "C" fn() -> *const ()>(b"hips_language\0")?;
        if entry().is_null() {
            bail!("grammar returned a null language");
        }
        Language::new(LanguageFn::from_raw(*entry))
    };
    Parser::new()
        .set_language(&language)
        .context("incompatible grammar ABI")?;
    Ok(LoadedGrammar {
        language,
        _library: library,
    })
}

fn download(url: &str, path: &Path) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(120))
        .build();
    let mut checksum = String::new();
    agent
        .get(&format!("{url}.sha256"))
        .call()?
        .into_reader()
        .take(1024)
        .read_to_string(&mut checksum)?;
    let expected = checksum
        .split_whitespace()
        .next()
        .context("empty grammar checksum")?;
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("invalid grammar checksum");
    }
    let dir = path.parent().context("grammar cache has no parent")?;
    std::fs::create_dir_all(dir)?;
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    let mut reader = agent
        .get(url)
        .call()?
        .into_reader()
        .take(128 * 1024 * 1024 + 1);
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut total = 0;
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total += n;
        if total > 128 * 1024 * 1024 {
            bail!("grammar exceeds size limit");
        }
        hash.update(&buffer[..n]);
        temp.write_all(&buffer[..n])?;
    }
    if format!("{:x}", hash.finalize()) != expected.to_ascii_lowercase() {
        bail!("grammar checksum mismatch");
    }
    temp.as_file().sync_all()?;
    // Validate before publishing. Unique temporary files + no-clobber publication
    // also allow independent hips processes to download the same grammar safely.
    drop(open(temp.path())?);
    match temp.persist_noclobber(path) {
        Ok(_) => Ok(()),
        Err(e) if e.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e.error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, thread};

    // Exercise real HTTP streaming, validation, dlopen, and cache publication.
    fn serve(checksum: String, body: Vec<u8>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/python", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            for response in [checksum.into_bytes(), body] {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response.len()
                )
                .unwrap();
                stream.write_all(&response).unwrap();
            }
        });
        (url, handle)
    }

    #[test]
    fn checksum_failure_leaves_no_cache_entry_or_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("python");
        let (url, server) = serve("0".repeat(64), b"corrupt library".to_vec());
        assert!(download(&url, &path)
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
        server.join().unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn invalid_library_is_not_published() {
        let dir = tempfile::tempdir().unwrap();
        let body = b"not a shared library".to_vec();
        let (url, server) = serve(format!("{:x}", Sha256::digest(&body)), body);
        assert!(download(&url, &dir.path().join("python")).is_err());
        server.join().unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn downloaded_grammar_parses_and_survives_a_concurrent_publication() {
        let source = PathBuf::from(
            std::env::var_os("HIPS_GRAMMAR_DIR")
                .expect("build grammars and set HIPS_GRAMMAR_DIR; see grammars/README.md"),
        )
        .join(format!(
            "libhips_grammar_python.{}",
            std::env::consts::DLL_EXTENSION
        ));
        let body = std::fs::read(source).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("python");
        let (url_a, server_a) = serve(format!("{:x}", Sha256::digest(&body)), body.clone());
        let (url_b, server_b) = serve(format!("{:x}", Sha256::digest(&body)), body);
        thread::scope(|scope| {
            let a = scope.spawn(|| download(&url_a, &path));
            let b = scope.spawn(|| download(&url_b, &path));
            a.join().unwrap().unwrap();
            b.join().unwrap().unwrap();
        });
        server_a.join().unwrap();
        server_b.join().unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        let grammar = open(&path).unwrap();
        let mut parser = Parser::new();
        parser.set_language(&grammar.language).unwrap();
        let tree = parser.parse("def hello():\n    pass\n", None).unwrap();
        assert!(!tree.root_node().has_error());
        assert!(tree.root_node().to_sexp().contains("function_definition"));
    }
}
