//! Dependency-free HTTP load generator for the search server.
//!
//! Opens N keep-alive connections, fires GET /search requests as fast as
//! the server answers, and reports QPS and latency percentiles.
//!
//! ```sh
//! cargo run --release -- serve --index ./index --addr 127.0.0.1:8080 &
//! cargo run --release --example http_load -- 127.0.0.1:8080 32 10 [queries.txt]
//! #                                            addr        conns seconds
//! ```
//! With a queries file (one query per line), those replace the built-in
//! query mix.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const QUERIES: &[&str] = &[
    "fn+main+rust",
    "import+json+python",
    "search+engine+index",
    "license+copyright+mit",
    "todo+fixme+hack",
    "async+await+tokio",
    "config+yaml+settings",
    "error+handling+exception",
    "git+commit+branch",
    "http+server+request+response",
];

fn worker(
    addr: &str,
    worker_id: usize,
    queries: &[String],
    stop: &AtomicBool,
    total: &AtomicU64,
    errors: &AtomicU64,
    latencies_us: &mut Vec<u64>,
) {
    let stream = TcpStream::connect(addr).expect("connect failed");
    stream.set_nodelay(true).ok();
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut stream = stream;
    let mut body = Vec::new();
    let mut line = String::new();
    let mut i = worker_id; // stagger query selection across workers

    while !stop.load(Ordering::Relaxed) {
        let query = &queries[i % queries.len()];
        i += 1;
        let request = format!(
            "GET /search?q={query}&k=10 HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n"
        );
        let start = Instant::now();
        if stream.write_all(request.as_bytes()).is_err() {
            errors.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Parse the status line + headers; find Content-Length.
        let mut content_length = 0usize;
        let mut ok = true;
        loop {
            line.clear();
            if reader.read_line(&mut line).is_err() || line.is_empty() {
                ok = false;
                break;
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(v) = trimmed
                .to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse::<usize>())
            {
                content_length = v.unwrap_or(0);
            }
        }
        if !ok {
            errors.fetch_add(1, Ordering::Relaxed);
            return;
        }
        body.resize(content_length, 0);
        if reader.read_exact(&mut body).is_err() {
            errors.fetch_add(1, Ordering::Relaxed);
            return;
        }
        latencies_us.push(start.elapsed().as_micros() as u64);
        total.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| "127.0.0.1:8080".into());
    let conns: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(32);
    let seconds: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(10);
    let queries: Vec<String> = match args.next() {
        Some(path) => std::fs::read_to_string(&path)
            .expect("failed to read queries file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().replace(' ', "+"))
            .collect(),
        None => QUERIES.iter().map(|q| q.to_string()).collect(),
    };
    assert!(!queries.is_empty(), "no queries");
    let queries = std::sync::Arc::new(queries);

    let stop = Arc::new(AtomicBool::new(false));
    let total = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let handles: Vec<_> = (0..conns)
        .map(|w| {
            let addr = addr.clone();
            let stop = Arc::clone(&stop);
            let total = Arc::clone(&total);
            let errors = Arc::clone(&errors);
            let queries = Arc::clone(&queries);
            std::thread::spawn(move || {
                let mut latencies = Vec::with_capacity(1 << 20);
                worker(&addr, w, &queries, &stop, &total, &errors, &mut latencies);
                latencies
            })
        })
        .collect();

    std::thread::sleep(Duration::from_secs(seconds));
    stop.store(true, Ordering::Relaxed);

    let mut latencies: Vec<u64> = Vec::new();
    for handle in handles {
        latencies.extend(handle.join().expect("worker panicked"));
    }
    let elapsed = start.elapsed().as_secs_f64();
    latencies.sort_unstable();
    let pct = |p: f64| -> f64 {
        if latencies.is_empty() {
            return 0.0;
        }
        let idx = ((latencies.len() as f64 * p).ceil() as usize).clamp(1, latencies.len()) - 1;
        latencies[idx] as f64 / 1000.0
    };

    let count = total.load(Ordering::Relaxed);
    println!("connections:   {conns}");
    println!("duration:      {elapsed:.1}s");
    println!("requests:      {count}");
    println!("errors:        {}", errors.load(Ordering::Relaxed));
    println!("throughput:    {:.0} req/s", count as f64 / elapsed);
    println!("latency p50:   {:.2} ms", pct(0.50));
    println!("latency p95:   {:.2} ms", pct(0.95));
    println!("latency p99:   {:.2} ms", pct(0.99));
}
