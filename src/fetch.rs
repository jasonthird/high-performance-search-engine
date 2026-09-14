//! Verified downloads for the release artifacts hips fetches on demand
//! (grammar libraries, the encoder's GGUF): a SHA-256 sidecar is read
//! first, the body is streamed through the hash under a size cap, the
//! caller validates the temporary file, and only then is it published
//! without clobbering, so concurrent processes fetching the same artifact
//! cannot corrupt each other.
use std::{
    io::{Read, Write},
    path::Path,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

pub(crate) fn download_verified(
    url: &str,
    path: &Path,
    max_bytes: u64,
    validate: impl Fn(&Path) -> Result<()>,
) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let mut checksum = String::new();
    agent
        .get(&format!("{url}.sha256"))
        .call()
        .with_context(|| format!("fetch {url}.sha256"))?
        .into_reader()
        .take(1024)
        .read_to_string(&mut checksum)?;
    let expected = checksum
        .split_whitespace()
        .next()
        .context("empty checksum file")?;
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("invalid checksum file for {url}");
    }
    let dir = path.parent().context("download target has no parent")?;
    std::fs::create_dir_all(dir)?;
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    let mut reader = agent
        .get(url)
        .call()
        .with_context(|| format!("fetch {url}"))?
        .into_reader()
        .take(max_bytes + 1);
    let mut hash = Sha256::new();
    let mut buffer = [0; 256 * 1024];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > max_bytes {
            bail!("{url} exceeds the size limit");
        }
        hash.update(&buffer[..n]);
        temp.write_all(&buffer[..n])?;
    }
    if format!("{:x}", hash.finalize()) != expected.to_ascii_lowercase() {
        bail!("checksum mismatch for {url}");
    }
    temp.as_file().sync_all()?;
    validate(temp.path())?;
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

    /// A one-shot HTTP server answering two requests: the checksum sidecar,
    /// then the body. Real streaming, real hashing, real publication.
    fn serve(responses: Vec<Vec<u8>>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/asset", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len()).unwrap();
                stream.write_all(&response).unwrap();
            }
        });
        (url, handle)
    }

    fn sha(body: &[u8]) -> String {
        format!("{:x}", Sha256::digest(body))
    }

    #[test]
    fn publishes_only_a_verified_and_validated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/asset.bin");
        let body = b"hello, artifact".to_vec();
        // The sidecar may carry a filename after the hash, like sha256sum's output.
        let (url, server) = serve(vec![format!("{}  asset.bin\n", sha(&body)).into_bytes(), body.clone()]);
        let validated = std::cell::Cell::new(false);
        download_verified(&url, &path, 1024, |p| {
            assert_eq!(std::fs::read(p).unwrap(), body);
            validated.set(true);
            Ok(())
        })
        .unwrap();
        server.join().unwrap();
        assert!(validated.get());
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(std::fs::read_dir(path.parent().unwrap()).unwrap().count(), 1, "no temp file left behind");
    }

    #[test]
    fn checksum_mismatch_and_invalid_sidecar_leave_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("asset.bin");
        let (url, server) = serve(vec!["0".repeat(64).into_bytes(), b"corrupt".to_vec()]);
        let err = download_verified(&url, &path, 1024, |_| Ok(())).unwrap_err().to_string();
        assert!(err.contains("checksum mismatch"), "{err}");
        server.join().unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);

        let (url, server) = serve(vec![b"not-a-hash".to_vec()]);
        let err = download_verified(&url, &path, 1024, |_| Ok(())).unwrap_err().to_string();
        assert!(err.contains("invalid checksum file"), "{err}");
        server.join().unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn rejected_validation_and_oversize_bodies_are_not_published() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("asset.bin");
        let body = b"looks fine to the hash".to_vec();
        let (url, server) = serve(vec![sha(&body).into_bytes(), body.clone()]);
        let err = download_verified(&url, &path, 1024, |_| anyhow::bail!("not the expected format")).unwrap_err();
        assert!(err.to_string().contains("not the expected format"));
        server.join().unwrap();
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);

        let big = vec![7u8; 2048];
        let (url, server) = serve(vec![sha(&big).into_bytes(), big]);
        let err = download_verified(&url, &path, 1024, |_| Ok(())).unwrap_err().to_string();
        assert!(err.contains("size limit"), "{err}");
        server.join().unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn concurrent_downloads_publish_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("asset.bin");
        let body = b"same bytes from two servers".to_vec();
        let (url_a, server_a) = serve(vec![sha(&body).into_bytes(), body.clone()]);
        let (url_b, server_b) = serve(vec![sha(&body).into_bytes(), body.clone()]);
        thread::scope(|scope| {
            let a = scope.spawn(|| download_verified(&url_a, &path, 1024, |_| Ok(())));
            let b = scope.spawn(|| download_verified(&url_b, &path, 1024, |_| Ok(())));
            a.join().unwrap().unwrap();
            b.join().unwrap().unwrap();
        });
        server_a.join().unwrap();
        server_b.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
